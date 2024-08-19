<div align="center">    
 
# SwiFT-UNETR

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

</div>


## 📌&nbsp;&nbsp;Introduction
This project is a continuing effort after [SwiFT](https://arxiv.org/abs/2307.05916) and [SwiFUN](https://www.biorxiv.org/content/10.1101/2024.05.29.596544v1.full.pdf), adapting SwiFUN to use SwiFT as an encoder model.

> Effective usage of this repository requires learning a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai). Knowledge of some experiment logging frameworks like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) is also recommended.

## 1. Description
This repository implements the SwiFUN (SwiFUN). 
- Our code offers the following things.
  - Trainer based on PyTorch Lightning for running the SwiFT integrated SwiFUN.
  - Data preprocessing/loading pipelines for 4D fMRI datasets.


## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/Padraig20/SwiFUN.git

# install project   
cd SwiFUN
conda env create -f envs/py39.yaml
conda activate py39
 ```

## 3. Architecture

The original SwiFUN adapted the input to use the channel dimension for the timepoints. This way, Swin could be used as an encoder. This project examines the possibility of using raw 4D fMRI data as the input of the encoder, which may or may not yield benefits in terms of model learning.

In order to fit the 4D data into the decoder U-NET architecture and to still remain computationally feasible, we chose to add an intermediate global average max pooling layer to squeeze the temporal dimension. This layer is able to learn the best representation of the 4D image among all the separate timepoints.

![image](https://github.com/user-attachments/assets/9a5390d4-b3f5-427b-849f-a457a83b95b1)


