import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Extract the row/col index from "tile_XX_YY.png"
# ------------------------------------------------------------
def parse_tile_position(tile_name):
    # example: "tile_03_14.png"
    base = os.path.splitext(tile_name)[0]
    _, r, c = base.split("_")
    return int(r), int(c)


# ------------------------------------------------------------
# 2. Encode an entire dataset (train/test)
# ------------------------------------------------------------
def encode_dataset(model, dataloader, device):
    model.eval()
    all_features = []
    tile_positions = []

    with torch.no_grad():
        for images, _, tile_names, _ in tqdm(dataloader, desc="Encoding dataset"):
            images = images.to(device)

            # encoded: (B, C, H', W')
            _, encoded = model(images, return_features=True)

            B, C, Hprime, Wprime = encoded.shape

            # flatten spatial positions
            flat = encoded.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            all_features.append(flat)

            # record tile coordinates
            for name in tile_names:
                tile_positions.append(parse_tile_position(name))

    features = np.concatenate(all_features, axis=0)
    return features, tile_positions, Hprime, Wprime


# ------------------------------------------------------------
# 3. Perform clustering
# ------------------------------------------------------------
def cluster_features(features, num_clusters=10):
    print("\nClustering features...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=4096,
        random_state=0,
        n_init="auto"
    )
    labels = kmeans.fit_predict(features)
    return labels, kmeans


# ------------------------------------------------------------
# 4. Build a stitched pseudo-segmentation mask
# ------------------------------------------------------------
def create_stitched_map(cluster_labels, tile_positions, Hprime, Wprime, tile_size=64):
    print("\nBuilding stitched segmentation map...")

    num_tiles = len(tile_positions)
    expected_points = num_tiles * Hprime * Wprime
    assert len(cluster_labels) == expected_points

    rows = [r for (r, c) in tile_positions]
    cols = [c for (r, c) in tile_positions]

    max_r = max(rows)
    max_c = max(cols)

    H_full = (max_r + 1) * tile_size
    W_full = (max_c + 1) * tile_size

    stitched = np.zeros((H_full, W_full), dtype=np.int32)

    idx = 0
    for (r, c) in tile_positions:

        tile_flat = cluster_labels[idx : idx + Hprime * Wprime]
        idx += Hprime * Wprime

        tile_map = tile_flat.reshape(Hprime, Wprime)

        tile_up = cv2.resize(tile_map, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

        y0, y1 = r * tile_size, (r + 1) * tile_size
        x0, x1 = c * tile_size, (c + 1) * tile_size

        stitched[y0:y1, x0:x1] = tile_up

    return stitched


# ------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------
def visualize_map(cluster_map, title="Cluster map"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cluster_map, cmap="tab20")
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# 6. Combine TRAIN + TEST stitched maps
# ------------------------------------------------------------
def combine_train_test_maps(stitched_train, stitched_test):
    # Both maps have the same size — combine them into one
    combined = np.where(stitched_train != 0, stitched_train, stitched_test)
    return combined

# ------------------------------------------------------------
# Combine TRAIN + VAL + TEST into a single global map
# ------------------------------------------------------------
def combine_three_maps(stitched_train, stitched_val, stitched_test):
    # Priority: train > val > test
    combined = np.where(stitched_train != 0, stitched_train,
                        np.where(stitched_val != 0, stitched_val, stitched_test))
    return combined


# ------------------------------------------------------------
# 7. MAIN PIPELINE
# ------------------------------------------------------------
def run_full_ssl_segmentation(model, train_loader, val_loader, test_loader, device, num_clusters=10):

    print("==== ENCODE TRAIN ====")
    train_features, train_pos, Hp, Wp = encode_dataset(model, train_loader, device)

    print("==== CLUSTER TRAIN ====")
    train_labels, kmeans = cluster_features(train_features, num_clusters=num_clusters)

    print("==== STITCH TRAIN ====")
    stitched_train = create_stitched_map(train_labels, train_pos, Hp, Wp)
    #visualize_map(stitched_train, "TRAIN: pseudo-segmentation")

    # -------------------------
    # VALIDATION
    # -------------------------
    print("==== ENCODE VAL ====")
    val_features, val_pos, _, _ = encode_dataset(model, val_loader, device)

    print("==== PREDICT VAL CLUSTERS ====")
    val_labels = kmeans.predict(val_features)

    print("==== STITCH VAL ====")
    stitched_val = create_stitched_map(val_labels, val_pos, Hp, Wp)
    #visualize_map(stitched_val, "VAL: pseudo-segmentation")

    # -------------------------
    # TEST
    # -------------------------
    print("==== ENCODE TEST ====")
    test_features, test_pos, _, _ = encode_dataset(model, test_loader, device)

    print("==== PREDICT TEST CLUSTERS ====")
    test_labels = kmeans.predict(test_features)

    print("==== STITCH TEST ====")
    stitched_test = create_stitched_map(test_labels, test_pos, Hp, Wp)
    #visualize_map(stitched_test, "TEST: pseudo-segmentation")

    # -------------------------
    # COMBINED MAP
    # -------------------------
    print("==== MERGE TRAIN + VAL + TEST ====")
    combined_map = combine_three_maps(stitched_train, stitched_val, stitched_test)
    visualize_map(combined_map, "FULL DATASET: train + val + test segmentation")

    return stitched_train, stitched_val, stitched_test, combined_map, kmeans