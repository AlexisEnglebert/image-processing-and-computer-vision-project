from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
import os
import re


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


# --------------------------------------------------------------------------------
# DISPLAY A GRID OF RANDOM IMAGES FROM A DATASET INSTANCE
# INPUTS: 
#     - dataset (Dataset): Instance of Dataset
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
# --------------------------------------------------------------------------------
def showDataset(dataset, param):

    rows, cols= 5, 3
    numPairs = rows*cols

    indices = random.sample(range(len(dataset)), numPairs)

    fig, axes = plt.subplots(rows, cols*2, figsize=(6,6), dpi=150)

    cmap = mcolors.ListedColormap([
        "black",   # 0 = "other"
        "blue",    # 1 = "water"
        "red",     # 2 = "building"
        "yellow",  # 3 = "farmland"
        "green"    # 4 = "forest"
    ])

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 2)

    for i, idx in enumerate(indices):
        image, mask, tileName, resizedImg = dataset[idx]

        row = i // cols
        col = i % cols

        # left subplot: image
        axes[row, col*2].imshow(resizedImg.numpy().astype('uint8').transpose(1,2,0))
        axes[row, col*2].set_title(tileName, fontsize=6)
        axes[row, col*2].axis('off')

        # right subplot: mask
        axes[row, col*2 + 1].imshow(mask.numpy().astype('uint8'), cmap=cmap, vmin=0, vmax=4)
        axes[row, col*2 + 1].set_title("mask", fontsize=6)
        axes[row, col*2 + 1].axis('off')

    plt.tight_layout
    plt.show()


# --------------------------------------------------------------------------------
# Reconstruction of full images from tiles and save results
# Inputs:
#     - allResizedImgs (np.ndarray): array of image tiles with shape (N, 3, H, W)
#     - allMasksPreds  (np.ndarray): array of predicted mask tiles with shape (N, H, W)
#     - allMasks       (np.ndarray): array of ground truth mask tiles with shape (N, H, W)
#     - allTileNames   (list of str): list of tile filenames, formatted as "tile_xx_yy.png"
# --------------------------------------------------------------------------------
def reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath):
   
    # Parse tile coordinates from filenames
    coords = []
    for fname in allTileNames:
        m = re.search(r"tile_(\d+)_(\d+)\.png", fname)
        if not m:
            raise ValueError(f"Filename {fname} does not match expected pattern")
        xx, yy = map(int, m.groups())
        coords.append((xx, yy))
    
    coords = np.array(coords)
    minX, minY = coords.min(axis=0)
    maxX, maxY = coords.max(axis=0)
    
    # Determine tile size and full image dimensions
    tileH, tileW = allResizedImgs.shape[2], allResizedImgs.shape[3]
    fullH = (maxX - minX + 1) * tileH
    fullW = (maxY - minY + 1) * tileW
    
    # Allocate empty arrays for reconstruction
    fullImg  = np.zeros((3, fullH, fullW), dtype=allResizedImgs.dtype)
    fullPred = np.zeros((fullH, fullW), dtype=allMasksPreds.dtype)
    fullMask  = np.zeros((fullH, fullW), dtype=allMasks.dtype)
    
    # Stitch image, predicted mask, and ground truth mask tiles
    for imgTile, predTile, maskTile, (xx, yy) in zip(allResizedImgs, allMasksPreds, allMasks, coords):
        i = xx - minX
        j = yy - minY
        y0, y1 = i * tileH, (i + 1) * tileH
        x0, x1 = j * tileW, (j + 1) * tileW
        
        fullImg[:, y0:y1, x0:x1] = imgTile
        fullPred[y0:y1, x0:x1]   = predTile
        fullMask[y0:y1, x0:x1]  = maskTile
    
    # Create overlay of predicted mask on original image
    imgRGB = np.transpose(fullImg, (1, 2, 0))
    overlay = imgRGB.copy()
    
    cmap = mcolors.ListedColormap([
        "black",   # 0 = "other"
        "blue",    # 1 = "water"
        "red",     # 2 = "building"
        "yellow",  # 3 = "farmland"
        "green"    # 4 = "forest"
    ])
    
    predColors = cmap(fullPred)[..., :3]
    alpha = 0.4
    overlay = (1 - alpha) * imgRGB / 255.0 + alpha * predColors

    # Save results
    createFolder(savePath)
    plt.imsave(os.path.join(savePath, "image.png"), imgRGB)
    plt.imsave(os.path.join(savePath, "prediction.png"), fullPred, cmap=cmap)
    plt.imsave(os.path.join(savePath, "mask.png"), fullMask, cmap=cmap) 
    plt.imsave(os.path.join(savePath, "overlay.png"), overlay)





    
    
