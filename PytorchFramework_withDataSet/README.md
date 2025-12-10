# Project LELEC2885

This project implements a pipeline for training a U-Net based model using Masked Autoencoders (MAE) as a proxy task for self-supervised pretraining, followed by clustering analysis of the learned features.

## Prerequisites

- Python 3.x
- PyTorch
- Dependencies listed in `requirements.txt`

## Running the proxy task (MAE)

The `main_proxy.py` script handles the training of the Masked Autoencoder.

### Basic usage

To train the MAE model using the parameters defined in `Todo_List/ProxyParameters.yaml`:

```bash
python main_proxy.py -exp ProxyParameters
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-exp <ExperimentName>` | Specifies the configuration file to use. Defaults to `DefaultExp`.|
| `--cluster` | Run clustering analysis on encoder features after training. |
| `--num_clusters <int>` | Number of clusters to learn (optional). |
| `--cluster_minibatch <int>` | Mini-batch size for clustering (optional). |
| `--save_features` | Persist the full encoded feature matrix to disk. |
| `--no_train` | Skip training and load existing weights. |

### Examples

Train a new model :
```bash
python main_proxy.py -exp ProxyParameters
```

Train a new model and run clustering analysis:
```bash
python main_proxy.py -exp ProxyParameters --cluster
```

Run clustering analysis on an already trained model (skip training):
```bash
python main_proxy.py -exp ProxyParameters --cluster --no_train
```

## Clustering analysis

The clustering analysis can also be run as a standalone script. It will take the weights of the model already trained.

### Standalone Usage

```bash
python clustering_analysis.py -exp <ExperimentName>
```

### Weight loading

The analysis script attempts to load model weights in the following order:
1.  **Current Experiment** : `Results/<ExperimentName>/_Weights/wghts.pkl`
2.  **Fallback** : `Results/ProxyParameters/_Weights/wghts.pkl`