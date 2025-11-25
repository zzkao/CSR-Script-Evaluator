#!/bin/bash

# Training
python train.py --name seg_uncertainty --train_all --batchsize 8 --gpu_ids 0 --num_epoch 1

# Inference / Demonstration
python demo.py --name seg_uncertainty --which_epoch 1

# Testing / Evaluation
python test.py --name seg_uncertainty --which_epoch 1
python evaluate.py --name seg_uncertainty --which_epoch 1
```