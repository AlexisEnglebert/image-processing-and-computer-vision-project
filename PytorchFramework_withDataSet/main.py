# This file is used to launch the training of the model for the proxy task

from Networks.model_proxy import *
from Dataset.makeGraph import *

import argparse
import yaml

import os
from os.path import dirname, abspath
from termcolor import colored

import clustering_analysis


rootDirectory    = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory,    "Dataset")
imgDirectory     = os.path.join(datasetDirectory, "images")
maskDirectory    = os.path.join(datasetDirectory, "annotations")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')
parser.add_argument('--cluster', action='store_true', help='Run clustering on encoder features after training.')
parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters to learn.')
parser.add_argument('--cluster_minibatch', type=int, default=None, help='Mini-batch size for clustering.')
parser.add_argument('--save_features', action='store_true', help='Persist the full encoded feature matrix to disk.')
parser.add_argument('--no_train', action='store_true', help='Skip training and load existing weights.')


######################################################################################
#
# MAIN PROCEDURE 
# launches an experiment whose parameters are described in a yaml file  
# 
# Example of use in the terminal: python main.py -exp DefaultExp
# with 'DefaultExp' beeing the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
######################################################################################

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open('Todo_List/' + parser.exp + '.yaml', 'r')
    param  = yaml.safe_load(stream)
    # Path to the folder that will contain results of the experiment 
    resultsPath = os.path.join(rootDirectory, "Results", parser.exp)

    myNetwork  = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    #showDataset(myNetwork.dataSetTrain, param)

    if not parser.no_train:
        print('Start to train the network')
        myNetwork.train()  # Train new MAE weights
        print('The network is trained')
    else:
        print('Skipping training, loading weights')
        myNetwork.loadWeights()
    
    myNetwork.evaluate()

    if parser.cluster:
        print(colored('Running clustering analysis...', 'cyan'))
        import clustering_analysis
        clustering_analysis.main(parser.exp)


if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)