#!/bin/bash

# Training
python main.py --dataset cora --encoder gat --decoder gat --seed 0 --max_epoch 1 --num_hidden 256 --num_layers 2 --lr 0.001 --drop_edge_rate 0.0 --mask_rate 0.5

# Inference / Demonstration
# Inference is included in the training script as evaluation

# Testing / Evaluation
# Evaluation is performed automatically during training on node classification task
```