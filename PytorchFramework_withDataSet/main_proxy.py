# This file is used to launch the training of the model for the proxy task

from Networks.model_proxy import *
from Dataset.makeGraph import *

import argparse
import yaml

import os
from os.path import dirname, abspath
from termcolor import colored


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


######################################################################################
#
# MAIN PROCEDURE 
# launches an experiment whose parameters are described in a yaml file  
# 
# Example of use in the terminal: python main_proxy.py -exp ProxyParameters
# with 'ProxyParameters' beeing the name of the yaml file (in the Todo_list folder) with 
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

    print(colored('Start to train the network', 'red'))

    # Uncomment this if you want to train !
    #myNetwork.train()

    print(colored('The network is trained', 'red'))
    
    myNetwork.loadWeights()
    
    # Uncomment this if you want to evaluate the model !
    #myNetwork.evaluate()

    cluster_cfg = param.get("CLUSTERING", {})
    run_clustering = parser.cluster or cluster_cfg.get("RUN", False)
    cluster_count = parser.num_clusters if parser.num_clusters is not None else cluster_cfg.get("NUM_CLUSTERS", 15)
    cluster_minibatch = (
        parser.cluster_minibatch if parser.cluster_minibatch is not None else cluster_cfg.get("MINIBATCH_SIZE", 1)
    )
    save_features = parser.save_features or cluster_cfg.get("SAVE_FEATURES", True)
    random_state = cluster_cfg.get("RANDOM_STATE", 0)

    if run_clustering:
        myNetwork.run_full_ssl_segmentation_(cluster_count, cluster_minibatch, random_state, save_features)
    
    

if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)
