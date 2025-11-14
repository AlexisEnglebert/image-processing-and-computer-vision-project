from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *
from Networks.Architectures.EncoderDecoderNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy
import tqdm

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim

from sklearn.metrics import jaccard_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


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
        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = EncoderDecoderNet(param).to(self.device)
        # -------------------
        # TRAINING PARAMETERS
        # -------------------

        # On doit changer les weigths car après entrainement il y en a bpc qui sont classé comme 0
        # Il faut donc pénaliser un peu plus le 0, vu qu'il correspond à la classe "unmapped area"
        class_weights = torch.tensor([0.8, 4.3, 3.5, 2.0, 2.6], device=self.device)

        weights_cfg = param["TRAINING"].get("CLASS_WEIGHTS")
        if weights_cfg is not None:
            class_weights = torch.tensor(weights_cfg, device=self.device)
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self.best_val_loss = float("inf")


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

    # -----------------------------------
    # TRAINING LOOP (dummy implementation)
    # -----------------------------------
    def train(self): 

        train_loss_history = []
        val_loss_history   = []

        # train for a given number of epochs
        for i in range(self.epoch):

            self.model.train(True)
            total_loss = 0.0
            for (images, masks, _, _) in tqdm.tqdm(self.trainDataLoader):
                images = images.to(self.device)
                masks  = masks.to(self.device, dtype=torch.long)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # backward pass + optimize
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            total_loss_epoch = total_loss / len(self.trainDataLoader)
            train_loss_history.append(total_loss_epoch)
            print(f"Loss at epoch {i}: {str(total_loss_epoch)}")

            # Validation 

            self.model.eval()

            with torch.no_grad():
                total_val_loss = 0.0
                for (images, masks, _, _) in self.valDataLoader:
                    images = images.to(self.device)
                    masks  = masks.to(self.device, dtype=torch.long)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

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
        plt.plot(range(1, self.epoch + 1), train_loss_history, label='Train Loss')
        plt.plot(range(1, self.epoch + 1), val_loss_history, label='Validation Loss')
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
        self.model.train(False)
        self.model.eval()
         
        allMasks, allMasksPreds, allTileNames, allResizedImgs = [], [], [], []
        for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
            images      = images.to(self.device)
            outputs     = self.model(images)

            images      = images.to('cpu')
            outputs     = outputs.to('cpu')

            masksPreds   = torch.argmax(outputs, dim=1)

            allMasks.extend(masks.data.numpy())
            allMasksPreds.extend(masksPreds.data.numpy())
            allResizedImgs.extend(resizedImgs.data.numpy())
            allTileNames.extend(tileNames)
        
        allMasks       = np.array(allMasks)
        allMasksPreds  = np.array(allMasksPreds)
        allResizedImgs = np.array(allResizedImgs)
            
        # Qualitative Evaluation
        savePath = os.path.join(self.resultsPath, "Test")
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath)
    
        # Quantitative Evaluation
        allMasks_flat      = allMasks.flatten()
        allMasksPreds_flat = allMasksPreds.flatten()

        # Compute metrics
        labels = list(range(5))
        mean_iou = jaccard_score(allMasks_flat, allMasksPreds_flat, average='macro', labels=labels)
        class_iou = jaccard_score(allMasks_flat, allMasksPreds_flat, average=None, labels=labels)

        # Calcule du F1-score pour chaque classe
        class_f1 = f1_score(allMasks_flat, allMasksPreds_flat, average=None, labels=labels)
        mean_f1 = float(np.mean(class_f1))

        # Matrice de confusion
        cm = confusion_matrix(allMasks_flat, allMasksPreds_flat, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0)


        print(f'Mean IoU: {mean_iou}')
        print(f'Class IoU: {class_iou}')
        print("------------------------")
        print(f'Mean class F1: {mean_f1}')
        print(f'Class F1: {class_f1}')
        print("------------------------")
        print(f"Best val Loss is: {self.best_val_loss}")
        graph = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
        graph.plot(cmap=plt.cm.Blues)
        plot_path = os.path.join(self.resultsPath, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.show()
    