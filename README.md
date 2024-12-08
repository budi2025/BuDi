# BuDi: Enhancing Bundle Recommendation via Bundle-Item Relations and User Individuality

This project is a pytorch implementation of the paper 'Enhancing Bundle Recommendation via Bundle-Item Relations and User Individuality'.
This project provides executable source code and preprocessed datasets used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.13.1

## Backbone
We use SASRec and BSARec as our backbone model.
1. **SASRec**: The implementation is adapted from the repository [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch). It is defined in `models.py`. For a transformer layer, we use the implementation of [rodosingh](https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580).
3. **BSARec**: It is available at [yehjin-shin/BSARec](https://github.com/yehjin-shin/BSARec).

## Datasets
We use 3 datasets in our work: Chess, Crypto, and Physics of StackExchange.
The preprocessed dataset is included in the repository: `./datset`.

## Repository Architecture
There are 2 folders and each consists of:
- dataset: preprocessed datasets
- src: source codes including models

## Running the code
You can train the model by following code:
```
python src/main.py
```

The different hyperparameters for each dataset are set in `main.py`. 

Alternatively, you can run the script `./shell/train.py`. 

