<div align="center">    
 
# SwiFT-BERT

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

</div>


## 📌&nbsp;&nbsp;Introduction
This project is a continuing effort after [SwiFT](https://arxiv.org/abs/2307.05916), using SwiFT for movie-related emotion prediction via the HBN dataset. We use several heads to further proceess the output of SwiFT, including a linear embedding, an MLP and a BERT model.

> Effective usage of this repository requires learning a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai). Knowledge of some experiment logging frameworks like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) is also recommended.

## 1. Description
This repository implements the SwiFT-BERT. 
- Our code offers the following things.
  - Trainer based on PyTorch Lightning for running the SwiFT integrated SwiFT-BERT.
  - Data preprocessing/loading pipelines for 4D fMRI datasets.

## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/Padraig20/SwiFT-BERT.git

# install project   
cd SwiFT-BERT
conda env create -f envs/py39.yaml
conda activate py39
 ```

## 3. Architecture

This architecture shows how BERT has been used on top of SwiFT. We squeeze the spatial dimensions including the channel dimensions, but retain the timepoints. In this way, we can make use of BERT's contextual understanding. This abstraction can be seen as similar to Named Entity Recognition (NER) from Natural Language Processing (NLP), where each timeframe is classified to be a certain entitiy.

![image](https://github.com/user-attachments/assets/3677acc3-cb0e-4657-af54-23c3bbd2b98c)



