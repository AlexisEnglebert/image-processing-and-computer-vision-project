import os
import re
import yaml
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Dataset.dataLoader import LLNDataset
from Networks.Architectures.EncoderDecoderNetworkProxy import EncoderDecoderNet
from Networks.Architectures.attentionunetProxy import attention_UNet as attention_UNet_Proxy

# Constants
GT_COLORS = ["black", "blue", "red", "yellow", "green"]
CLASS_NAMES = ['Unmapped', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
GT_CMAP = mcolors.ListedColormap(GT_COLORS)

def get_cluster_colors(n_clusters):
    # Generate distinct colors for clusters
    return plt.cm.tab20(np.linspace(0, 1, n_clusters))[:, :3]

def get_cluster_cmap(n_clusters):
    # Generate a colormap for clusters
    return mcolors.ListedColormap(get_cluster_colors(n_clusters))

# Model wrappers to extract only encoder features
class EncoderExtractor(nn.Module):
    # Wrapper to extract only encoder features from EncoderDecoderNet
    def __init__(self, model):
        super().__init__()
        self.layers = nn.Sequential(
            model.encoder1, model.pool,
            model.encoder2, model.pool,
            model.encoder3, model.pool,
            model.bottleneck
        )
    
    def forward(self, x):
        return self.layers(x)

class AttentionEncoderExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.e1 = model.e1
        self.e2 = model.e2
        self.e3 = model.e3
        self.b1 = model.b1
    
    def forward(self, x):
        _, p1 = self.e1(x)
        _, p2 = self.e2(p1)
        _, p3 = self.e3(p2)
        return self.b1(p3)

# Detect model architecture from weights file
def detect_model_type(weights_path):
    state_dict = torch.load(weights_path, weights_only=True)
    keys = list(state_dict.keys())

    for k in keys:
        if k.startswith('e1.') or k.startswith('d1.ag.'):
            return 'attention_unet'

    for k in keys:
        if k.startswith('encoder1.'):
            return 'encoder_decoder'
            
    raise ValueError("Unknown model architecture.")

# Feature extraction and clustering
def extract_features(encoder, dataloader, device):
    encoder.eval()
    all_features = []
    tile_info = []
    
    with torch.no_grad():
        for images, masks, tile_names, _ in dataloader:
            images = images.to(device)
            features = encoder(images)
            B, C, H_prime, W_prime = features.shape
            
            # Flatten : (B, C, H', W') -> (B*H'*W', C)
            flat = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            all_features.append(flat)
            
            # Store tile info
            for i in range(len(tile_names)):
                name = tile_names[i]
                mask = masks[i].numpy()
                tile_info.append((name, H_prime, W_prime, mask))
    
    return np.concatenate(all_features, axis=0), tile_info

def apply_clustering_to_tiles(encoder, dataloader, kmeans, scaler, device, original_size):
    encoder.eval()
    pseudo_masks = []
    gt_masks = []
    tile_names = []
    
    with torch.no_grad():
        for images, masks, names, _ in dataloader:
            images = images.to(device)
            features = encoder(images)
            B, C, Hp, Wp = features.shape
            
            for i in range(B):
                # Extract features for this tile
                tile_feat = features[i].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
                
                # Normalize
                tile_feat = scaler.transform(tile_feat)
                
                # Predict clusters
                cluster_map = kmeans.predict(tile_feat).reshape(Hp, Wp)
                
                # Upscale to original size
                upscaled = cv2.resize(
                    cluster_map.astype(np.float32), 
                    (original_size[1], original_size[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                
                pseudo_masks.append(upscaled)
                gt_masks.append(masks[i].numpy())
                tile_names.append(names[i])
    
    return pseudo_masks, gt_masks, tile_names

def analyze_cluster_correspondence(pseudo_masks, gt_masks, n_clusters, n_classes=5):
    # Analyze correspondence between clusters and ground truth classes
    
    # Flatten all masks into one big array
    pseudo_flat = []
    for m in pseudo_masks:
        pseudo_flat.append(m.flatten())
    pseudo_flat = np.concatenate(pseudo_flat)
    
    gt_flat = []
    for m in gt_masks:
        gt_flat.append(m.flatten())
    gt_flat = np.concatenate(gt_flat)
    
    correspondence = np.zeros((n_clusters, n_classes))
    
    for cluster_id in range(n_clusters):
        mask = (pseudo_flat == cluster_id)
        
        if np.any(mask):
            gt_values = gt_flat[mask]
            
            counts = np.bincount(gt_values, minlength=n_classes)
            correspondence[cluster_id] = counts
    
    # Normalize rows to get percentages
    correspondence_norm = np.zeros_like(correspondence)
    row_sums = correspondence.sum(axis=1)
    
    for i in range(n_clusters):
        if row_sums[i] > 0:
            correspondence_norm[i] = correspondence[i] / row_sums[i]
    
    return correspondence, correspondence_norm

# Visualization


def plot_correspondence_matrix(correspondence_norm, n_clusters, save_path):
    # Plot heatmap of cluster-to-class correspondence
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correspondence_norm, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(n_clusters))
    
    yticklabels = []
    for i in range(n_clusters):
        yticklabels.append('Cluster ' + str(i))
    ax.set_yticklabels(yticklabels)
    
    ax.set_xlabel('Ground truth class')
    ax.set_ylabel('Cluster')
    ax.set_title('Cluster to ground truth class correspondence')
    
    for i in range(n_clusters):
        for j in range(5):
            val = correspondence_norm[i, j]
            color = 'black'
            if val >= 0.5:
                color = 'white'
            
            text = "{:.2f}".format(val)
            ax.text(j, i, text, ha='center', va='center', color=color)
    
    plt.colorbar(im, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()





def reconstruct_full_image_clustering(encoder, dataloader, kmeans, scaler, device, 
                                       original_size, n_clusters, save_dir):
    # Reconstruct full image from tiles
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    encoder.eval()
    
    tiles = []
    with torch.no_grad():
        for images, masks, names, resized_imgs in dataloader:
            images = images.to(device)
            features = encoder(images)
            B, C, Hp, Wp = features.shape
            
            for i in range(B):
                # Clustering
                tile_feat = features[i].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
                tile_feat_norm = scaler.transform(tile_feat)
                cluster_map = kmeans.predict(tile_feat_norm).reshape(Hp, Wp)
                
                upscaled = cv2.resize(
                    cluster_map.astype(np.float32),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                
                # Parse coordinates
                # Expecting format like "tile_12_34.png"
                m = re.search(r"tile_(\d+)_(\d+)\.png", names[i])
                if m:
                    r = int(m.group(1))
                    c = int(m.group(2))
                    
                    tile_data = {
                        'img': resized_imgs[i].numpy().transpose(1, 2, 0).astype(np.uint8),
                        'mask': masks[i].numpy(),
                        'pred': upscaled,
                        'r': r,
                        'c': c
                    }
                    tiles.append(tile_data)

    if len(tiles) == 0:
        return

    # Determine dimensions
    rows = []
    cols = []
    for t in tiles:
        rows.append(t['r'])
        cols.append(t['c'])
        
    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)
    
    h, w, _ = tiles[0]['img'].shape
    
    full_h = (max_r - min_r + 1) * h
    full_w = (max_c - min_c + 1) * w
    
    full_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    full_mask = np.zeros((full_h, full_w), dtype=int)
    full_pred = np.zeros((full_h, full_w), dtype=int)
    
    for t in tiles:
        r = t['r'] - min_r
        c = t['c'] - min_c
        
        y_start = r * h
        y_end = (r + 1) * h
        x_start = c * w
        x_end = (c + 1) * w
        
        full_img[y_start:y_end, x_start:x_end] = t['img']
        full_mask[y_start:y_end, x_start:x_end] = t['mask']
        full_pred[y_start:y_end, x_start:x_end] = t['pred']
    
    # Visualization
    cluster_colors = get_cluster_colors(n_clusters)
    pred_rgb = cluster_colors[full_pred]
    
    overlay = 0.6 * full_img / 255.0 + 0.4 * pred_rgb
    overlay = np.clip(overlay, 0, 1)
    
    plt.imsave(os.path.join(save_dir, "full_image.png"), full_img)
    plt.imsave(os.path.join(save_dir, "full_gt_mask.png"), full_mask, cmap=GT_CMAP, vmin=0, vmax=4)
    plt.imsave(os.path.join(save_dir, "full_overlay.png"), overlay)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(full_pred, cmap=get_cluster_cmap(n_clusters), vmin=0, vmax=n_clusters-1)
    plt.colorbar(im, ax=ax, ticks=range(n_clusters), label='Cluster')
    ax.axis('off')
    plt.savefig(os.path.join(save_dir, "full_clustering_prediction.png"), dpi=200)
    plt.close()

def main(experiment_name="ProxyParameters"):
    # Get the directory where this script is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load config for the experiment
    config_path = os.path.join(root_dir, "Todo_List", f"{experiment_name}.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(root_dir, "Todo_List", "DefaultExp.yaml")

    # Logic for weights
    weights_path = os.path.join(root_dir, "Results", experiment_name, "_Weights", "wghts.pkl")
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        weights_path = os.path.join(root_dir, "Results", "ProxyParameters", "_Weights", "wghts.pkl")
        print(f"Using fallback weights: {weights_path}")
        
    save_dir = os.path.join(root_dir, "Results", "ClusteringAnalysis")
    
    # Load configuration
    with open(config_path, 'r') as f:
        param = yaml.safe_load(f)
    
    device = param["TRAINING"]["DEVICE"]
    
    # Parse resize shape "256x256" -> 256, 256
    shape_str = param["DATASET"]["RESIZE_SHAPE"]
    parts = shape_str.split("x")
    h = int(parts[0])
    w = int(parts[1])
    original_size = (h, w)
    
    print("=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load Model
    if not os.path.exists(weights_path):
        print("Weights not found at " + weights_path)
        return

    model_type = detect_model_type(weights_path)
    print("Model: " + model_type)
    
    if model_type == 'attention_unet':
        model = attention_UNet_Proxy(param).to(device)
        encoder_cls = AttentionEncoderExtractor
    else:
        model = EncoderDecoderNet(param).to(device)
        encoder_cls = EncoderExtractor
        
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    encoder = encoder_cls(model).to(device)
    
    # Data
    dataset_dir = os.path.join(root_dir, "Dataset")
    img_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "annotations")
    
    train_dataset = LLNDataset(img_dir, mask_dir, "train", param)
    test_dataset = LLNDataset(img_dir, mask_dir, "test", param)
    
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)
    
    # Analysis
    print("Extracting features...")
    train_features, _ = extract_features(encoder, train_loader, device)
    
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    
    cluster_counts = [5, 8, 10, 15]
    
    for n_clusters in cluster_counts:
        print("")
        print("--- " + str(n_clusters) + " Clusters ---")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(train_features_norm)
        
        pseudo_masks, gt_masks, tile_names = apply_clustering_to_tiles(
            encoder, test_loader, kmeans, scaler, device, original_size
        )
        
        _, corr_norm = analyze_cluster_correspondence(pseudo_masks, gt_masks, n_clusters)
        
        # Print stats
        header = "{:<10} ".format("Cluster")
        for name in CLASS_NAMES:
            header += "{:>10} ".format(name)
        print(header)
        
        for i in range(n_clusters):
            row_str = "Cluster {:<2} ".format(i)
            for j in range(5):
                val = corr_norm[i, j] * 100
                row_str += "{:>9.1f}% ".format(val)
            print(row_str)
            
        # Visualize
        cluster_dir = os.path.join(save_dir, "clusters_" + str(n_clusters))
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        plot_path = os.path.join(cluster_dir, "correspondence.png")
        plot_correspondence_matrix(corr_norm, n_clusters, plot_path)
        
        recon_dir = os.path.join(cluster_dir, "full_reconstruction")
        reconstruct_full_image_clustering(encoder, test_loader, kmeans, scaler, device, original_size, n_clusters, recon_dir)

    print("")
    print("Done. Results in " + save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=str, default='ProxyParameters', help='Experiment name')
    args = parser.parse_args()
    main(args.exp)