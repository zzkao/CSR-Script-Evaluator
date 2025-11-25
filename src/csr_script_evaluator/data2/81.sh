#!/bin/bash

# Training
python train.py --cfg config/imbalance/VLCS_lt.yaml --exp_name vlcs_resnet50_lt --trainer RIDE --seed 0 --num_epochs 1 --batch_size 32

# Inference / Demonstration
# No specific inference commands in README

# Testing / Evaluation
python test.py --cfg config/imbalance/VLCS_lt.yaml --exp_name vlcs_resnet50_lt --seed 0
```