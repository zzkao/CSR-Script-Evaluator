#!/bin/bash

# Training
python cifar_train.py --gpu 0 --dataset cifar10 --loss_type CE --train_rule None --imb_factor 0.01 --epochs 1

# Inference / Demonstration
# No specific inference commands in README

# Testing / Evaluation
# Evaluation is performed during training
```