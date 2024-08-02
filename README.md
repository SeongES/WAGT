# WAGT
This repository contains the code of 
[WAGT] Parameter Efficient Tuning for Graph Neural Networks via a Weight Adaptive Module.

## Environment Setup

```
conda create -n wagt python=3.7
conda activate wagt

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==1.7.2
pip install wandb
pip install rdkit
pip install PrettyTable
```

## Pre-trained Models
We use the pre-trained models from paper [*Strategies for Pre-training Graph Neural Networks*](https://github.com/snap-stanford/pretrain-gnns) and 
[*Simgrace*](https://github.com/mpanpan/SimGRACE). Please refer to the linked Git repositories for detail.

The pre-trained checkpoints that we use are:
```infomax.pth masking.pth contextpred.pth edgepred.pth simgrace.pth```

To finetune other pre-trained models, add the checkpoint file under ```./bio/model_gin``` and ```./chem/model_gin```.

## Dataset
The datasets for pre-training and downstream tasks utilized in our experiments are from the paper *Strategies for Pre-training Graph Neural Networks*.
The biology (2.5GB) and chemistry (2GB) datasets can be accessed [here](https://github.com/snap-stanford/pretrain-gnns).
Download each domain's dataset and unzip under `./bio/dataset` and `./chem/dataset`

## Run WAGT
Biology Dataset
```
cd bio
bash run.sh
```
Chemistry Dataset
```
cd chem
bash run.sh
```
