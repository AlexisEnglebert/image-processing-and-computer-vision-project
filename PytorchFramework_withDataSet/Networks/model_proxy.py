from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.EncoderDecoderNetworkProxy import *

import numpy as np
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

    def collect_encoded_features(self):
        self.model.eval()

        all_features = []
        h_prime, w_prime, channels = None, None, None

        with torch.no_grad():
            for (images, _, _, _) in tqdm.tqdm(trainDataLoader, desc="Collecting encoded features"):
                images = images.to(self.device)
                _, encoded = self.model(images, return_features=True)
                batch_count, channels, h_prime, w_prime = encoded.shape
                flattened = encoded.permute(0, 2, 3, 1).reshape(-1, channels).cpu()
                all_features.append(flattened)

        if not all_features:
            return np.empty((0, 0))

        features = torch.cat(all_features, dim=0).numpy()
  
        return features

    def cluster_training_features(self, num_clusters=10, minibatch_size=4096, random_state=0, save_features=False ):
    
        features  = self.collect_encoded_features()
        if features.shape[0] == 0:
            raise RuntimeError("No encoded features collected from the training set.")

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

                    outputs = self.model(x_masked)
                    # Compute the loss only on the masked pixels
                    loss = self.criterion(outputs[mask.expand_as(outputs)], images[mask.expand_as(images)])

                    total_val_loss += loss.item()

                total_val_loss_epoch = total_val_loss / len(self.valDataLoader)
                val_loss_history.append(total_val_loss_epoch)
                print(f"Validation Loss: ", str(total_val_loss_epoch))

                if total_val_loss < self.best_val_loss:
                    self.best_weights = copy.deepcopy(self.model.state_dict())
                    self.best_val_loss = total_val_loss


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

                # Reconstruct
                output = self.model(images_masked)

                # Compute the loss only on the masked pixels
                loss = self.criterion(output[mask.expand_as(output)], images[mask.expand_as(images)])

                total_loss += loss.item() * images.size(0)
                count += images.size(0)
        # Print the loss of the model on test data
        print("Loss: " + str(total_loss / count))
        return total_loss / count
