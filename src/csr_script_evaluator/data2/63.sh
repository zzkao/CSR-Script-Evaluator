#!/bin/bash

# Training
python preprocess_OAG.py --data_dir ./dataset/oag_output --cuda 0
python train_RGCN.py --model_dir ./model_RGCN --n_epoch 1 --n_batch 32 --cuda 0
python train_HGT.py --model_dir ./model_HGT --n_epoch 1 --n_batch 32 --cuda 0
python train_GPTGNN.py --data_dir ./dataset/oag_output --model_dir ./model_GPTGNN --n_epoch 1 --n_batch 32 --cuda 0

# Inference / Demonstration
# No specific inference commands in README

# Testing / Evaluation
# Evaluation is performed during training
```