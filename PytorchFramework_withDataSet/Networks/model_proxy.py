from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.EncoderDecoderNetworkProxy import *
from Networks.Architectures.attentionunetProxy import *
from Networks.ssl_clustering import run_full_ssl_segmentation

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2885)
import os
import copy
import tqdm
import json

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim

from sklearn.metrics import jaccard_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)

# Function that randomly selects the patches to be masked
def apply_mask(images, mask_ratio=0.7, patch_size=8, replace_with_noise=True):
    """
    Block (patch) masking: split images into non-overlapping patches of patch_size and randomly mask patches.
    images: (B,C,H,W), H and W must be divisible by patch_size
    returns: images_masked, mask_bool
    mask_bool is True for masked pixels
    """
    B, C, H, W = np.shape(images)
    # verify the height and width of image are multiples of the patch size
    assert H % patch_size == 0 and W % patch_size == 0
    # coordinates of each patch
    gh, gw = H // patch_size, W // patch_size
    # create mask for patches
    patch_mask = torch.rand(B, gh, gw, device=images.device) < mask_ratio
    # expand to pixel mask
    patch_mask = patch_mask.unsqueeze(-1).unsqueeze(-1)  # B,gh,gw,1,1
    patch_mask = patch_mask.expand(-1, -1, -1, patch_size, patch_size)  # B,gh,gw,ps,ps
    mask_bool = patch_mask.reshape(B, 1, H, W)  # B,1,H,W

    images_masked = images.clone()
    # Mask using noise
    if replace_with_noise:
        noise = torch.randn_like(images) * 0.1 + 0.5
        images_masked[mask_bool.expand_as(images_masked)] = noise[mask_bool.expand_as(images_masked)]
    # Mask using 0
    else:
        images_masked[mask_bool.expand_as(images_masked)] = 0.0

    return images_masked, mask_bool

def apply_mask_patch(x, mask_ratio=0.75, patch_size=8):
    """
    Apply patch-based masking for MAE.
    """
    B, C, H, W = x.shape
    
    # Number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Create patch-level mask
    patch_mask = torch.rand(B, 1, num_patches_h, num_patches_w, device=x.device) < mask_ratio
    
    # Upsample to pixel level
    mask = patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    
    x_masked = x.clone()
    x_masked[mask.expand_as(x)] = 0
    
    return x_masked, mask

######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]
        self.weight_decay  = param["TRAINING"]["WEIGHT_DECAY"]
        self.patience      = param["TRAINING"].get("PATIENCE", 10)
        self.grad_clip     = param["TRAINING"].get("GRAD_CLIP", 1.0)
        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = EncoderDecoderNet(param).to(self.device)
        #self.model = attention_UNet(param).to(self.device)
        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        # Use MSE loss for critetion
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=5, factor=0.5)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=2)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=2)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=2)
    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))
    
    def run_full_ssl_segmentation_(self):
        run_full_ssl_segmentation(
        model=self.model,
        train_loader=self.trainDataLoader,
        val_loader=self.valDataLoader,
        test_loader=self.testDataLoader,
        device=self.device,
        num_clusters=15
        )
    
    def encode_images(self, images):
        """Return the encoder feature maps (not flattened)."""
        with torch.no_grad():
            images = images.to(self.device)
            _, encoded = self.model(images, return_features=True)
        return encoded   # shape (B, C, H', W')

    import re

    def stitch_full_map(self, cluster_maps, tile_names, tile_H, tile_W):
        """
        Rebuild the full map from individual clustered tiles.

        cluster_maps: list or array of (H, W) cluster predictions per tile
        tile_names: list of filenames like "tile_03_07.png"
        tile_H, tile_W: size of each tile in pixels

        Returns:
            full_map: (full_H, full_W) array with cluster labels
        """

        # Extract row/col indices from tile names
        coords = []
        for name in tile_names:
            match = re.findall(r"tile_(\d+)_(\d+)", name)
            if not match:
                raise RuntimeError(f"Cannot extract row/col from file name: {name}")
            row, col = map(int, match[0])
            coords.append((row, col))

        # Get grid size
        max_row = max(r for r, _ in coords)
        max_col = max(c for _, c in coords)
        grid_rows = max_row + 1
        grid_cols = max_col + 1

        # Allocate the final giant map
        full_H = grid_rows * tile_H
        full_W = grid_cols * tile_W
        full_map = np.zeros((full_H, full_W), dtype=np.int32)

        # Fill the giant map
        for (r, c), tile in zip(coords, cluster_maps):
            y0 = r * tile_H
            y1 = y0 + tile_H
            x0 = c * tile_W
            x1 = x0 + tile_W
            full_map[y0:y1, x0:x1] = tile

        return full_map


    def cluster_test_features_and_reconstruct(self, clustering, save_output=True):
        """
        Runs the trained KMeans clustering on the encoded pixels of the test images.
        Reconstructs the pseudo-segmentation maps by:
        1. Encoding test images
        2. Predicting cluster labels per encoded pixel
        3. Reshaping (H', W')
        4. Upsampling back to (H, W)
        5. Stitching into a full scene
        """

        self.model.eval()

        cluster_dir = os.path.join(self.resultsPath, "_Clusters")
        createFolder(cluster_dir)

        stitched_maps = []     # one map per test image
        stitched_gt = []       # store ground truth masks

        with torch.no_grad():
            for (images, _, _, masks) in tqdm.tqdm(self.testDataLoader, desc="Clustering test data"):
                B, C, H, W = images.shape

                encoded = self.encode_images(images)
                _, Cenc, Hp, Wp = encoded.shape

                # Flatten encoded features
                flat = encoded.permute(0,2,3,1).reshape(-1, Cenc).cpu().numpy()

                # Predict clusters
                flat_labels = clustering.predict(flat)

                # Reshape back to (B, Hp, Wp)
                cluster_maps = flat_labels.reshape(B, Hp, Wp)

                # Upscale from (Hp,Wp) → (H, W)
                cluster_maps_up = torch.nn.functional.interpolate(
                    torch.tensor(cluster_maps).unsqueeze(1).float(),
                    size=(H, W),
                    mode="nearest"
                ).squeeze(1).numpy()

                # Store maps
                stitched_maps.extend(cluster_maps_up)
                stitched_gt.extend(masks.numpy())

        stitched_maps = np.array(stitched_maps)
        stitched_gt = np.array(stitched_gt)

        # Save output if needed
        if save_output:
            np.save(os.path.join(cluster_dir, "test_cluster_maps.npy"), stitched_maps)
            np.save(os.path.join(cluster_dir, "test_ground_truth.npy"), stitched_gt)
            print(f"Saved clustered maps to {cluster_dir}")

        return stitched_maps, stitched_gt

    def visualize_cluster_map(self, cluster_map, title="Cluster Map"):
        plt.figure(figsize=(6,6))
        plt.imshow(cluster_map, cmap="tab20")
        plt.title(title)
        plt.axis("off")
        plt.show()

    def collect_encoded_features(self):
        self.model.eval()

        all_features = []
        h_prime, w_prime, channels = None, None, None

        with torch.no_grad():
            for (images, _, _, _) in tqdm.tqdm(self.trainDataLoader, desc="Collecting encoded features"):
                images = images.to(self.device)
                _, encoded = self.model(images, return_features=True)
                batch_count, channels, h_prime, w_prime = encoded.shape
                flattened = encoded.permute(0, 2, 3, 1).reshape(-1, channels).cpu()
                all_features.append(flattened)

        if not all_features:
            return np.empty((0, 0))

        features = torch.cat(all_features, dim=0).numpy()
  
        return features

    def cluster_training_features(self, num_clusters=5, minibatch_size=64, random_state=0, save_features=False ):
    
        features  = self.collect_encoded_features()
        if features.shape[0] == 0:
            raise RuntimeError("No encoded features collected from the training set.")
        
        #pca = PCA(n_components=50)
        #features = pca.fit_transform(features)

        clustering = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=minibatch_size,
            random_state=random_state,
            n_init="auto",
        )
        cluster_labels = clustering.fit_predict(features)

        cluster_dir = os.path.join(self.resultsPath, "_Clusters")
        createFolder(cluster_dir)

        np.save(os.path.join(cluster_dir, "cluster_labels.npy"), cluster_labels)
        np.save(os.path.join(cluster_dir, "cluster_centers.npy"), clustering.cluster_centers_)

        if save_features:
            np.save(os.path.join(cluster_dir, "encoded_features.npy"), features)

        print( f"Artifacts saved to {cluster_dir}.")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(dict(zip(unique, counts)))
        return cluster_labels, clustering

    # -----------------------------------
    # TRAINING LOOP (dummy implementation)
    # -----------------------------------
    def train(self): 

        train_loss_history = []
        val_loss_history   = []
        warmup_epochs = 5
        base_lr = self.lr

        # train for a given number of epochs
        for i in range(self.epoch):
            
            # Learning rate warmup
            if i < warmup_epochs:
                lr = base_lr * (i + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            self.model.train(True)
            total_loss = 0.0
            for (images, _, _, _) in tqdm.tqdm(self.trainDataLoader):
                images = images.to(self.device)

                # Apply random masking
                x_masked, mask = apply_mask(images, mask_ratio=0.70)
                #x_masked, mask = apply_mask_patch(images, mask_ratio=0.75, patch_size=16)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(x_masked)
                # Compute the loss only on the masked pixels
                loss = self.criterion(outputs[mask.expand_as(outputs)], images[mask.expand_as(images)])

                # backward pass + optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                total_loss += loss.item()

            total_loss_epoch = total_loss / len(self.trainDataLoader)
            train_loss_history.append(total_loss_epoch)
            print(f"Loss at epoch {i}: {str(total_loss_epoch)}")

            # Validation 

            self.model.eval()

            with torch.no_grad():
                total_val_loss = 0.0
                for (images, _, _, _) in self.valDataLoader:
                    images = images.to(self.device)

                    # Apply random masking
                    x_masked, mask = apply_mask(images, mask_ratio=0.70)
                    #x_masked, mask = apply_mask_patch(images, mask_ratio=0.75, patch_size=16)

                    outputs = self.model(x_masked)
                    # Compute the loss only on the masked pixels
                    loss = self.criterion(outputs[mask.expand_as(outputs)], images[mask.expand_as(images)])

                    total_val_loss += loss.item()

                total_val_loss_epoch = total_val_loss / len(self.valDataLoader)
                val_loss_history.append(total_val_loss_epoch)
                print(f"Validation Loss: ", str(total_val_loss_epoch))

                if total_val_loss_epoch < self.best_val_loss:
                    self.best_weights = copy.deepcopy(self.model.state_dict())
                    self.best_val_loss = total_val_loss_epoch


        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.resultsPath, 'loss_curves.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss curves graph saved to {plot_path}")


        # Save the model weights
        
        torch.save(self.best_weights, wghtsPath + '/wghts.pkl') 



    # -------------------------------------------------
    # EVALUATION PROCEDURE (dummy implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for (images, _, _, _) in self.testDataLoader:
                images = images.to(self.device)
                
                # Apply mask
                images_masked, mask = apply_mask(images, mask_ratio=0.70)
                #images_masked, mask = apply_mask_patch(images, mask_ratio=0.75, patch_size=16)

                # Reconstruct
                output = self.model(images_masked)

                # Compute the loss only on the masked pixels
                loss = self.criterion(output[mask.expand_as(output)], images[mask.expand_as(images)])

                total_loss += loss.item() * images.size(0)
                count += images.size(0)
        # Print the loss of the model on test data
        print("Loss: " + str(total_loss / count))
        return total_loss / count
