# Project LELEC2885

This project implements a pipeline for training a U-Net based model using Masked Autoencoders (MAE) as a proxy task for self-supervised pretraining, followed by clustering analysis of the learned features.

## Prerequisites

- Python 3.x
- PyTorch
- Dependencies listed in `requirements.txt`

## Running the proxy task (MAE)

The `main_proxy.py` script handles the training of the Masked Autoencoder and the subsequent clustering analysis.

### Basic usage

To train the MAE model using the parameters defined in `Todo_List/ProxyParameters.yaml`:

```bash
python main_proxy.py -exp ProxyParameters
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-exp <ExperimentName>` | Specifies the configuration file to use. Defaults to `DefaultExp`.|
| `--cluster` | Runs the clustering analysis script (`clustering_analysis.py`) after training or loading the model. |
| `--no_train` | Skips the training phase and loads pre-trained weights. Useful for running analysis on an existing model. |

### Examples

Train a new model and run clustering analysis :
```bash
python main_proxy.py -exp ProxyParameters --cluster
```

Run clustering analysis on an already trained model (skip training) :
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