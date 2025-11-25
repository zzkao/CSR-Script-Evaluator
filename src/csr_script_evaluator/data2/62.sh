#!/bin/bash

# Training
python train_GCN.py --dataset PROTEINS --hidden_dim 16 --learning_rate 0.01 --num_epochs 1
python train_GIN.py --dataset PROTEINS --hidden_dim 16 --learning_rate 0.01 --num_epochs 1

# Inference / Demonstration
# No specific inference commands in README

# Testing / Evaluation
python train_GCN.py --dataset PROTEINS --hidden_dim 16 --learning_rate 0.01 --num_epochs 1
python train_GIN.py --dataset PROTEINS --hidden_dim 16 --learning_rate 0.01 --num_epochs 1
python train_GT.py --dataset PROTEINS --hidden_dim 16 --learning_rate 0.01 --num_epochs 1
```