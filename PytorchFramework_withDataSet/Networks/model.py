from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim


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

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = Net(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = ...
        self.optimizer = ... 

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))

    # -----------------------------------
    # TRAINING LOOP (dummy implementation)
    # -----------------------------------
    def train(self): 
        # train for a given number of epochs
        for i in range(self.epoch):
            print("Loss at i-th epoch: ", str(np.random.random_sample()))
            modelWts = copy.deepcopy(self.model.state_dict())

        # Print learning curves
        # Implement this...

        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')



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
        # Implement this ! 

