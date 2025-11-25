#!/bin/bash

# Training
python train.py --data_dir data --dataset agedb-dir --reweight sqrt_inv --loss l1 --epoch 1

# Inference / Demonstration
# No explicit inference commands in README

# Testing / Evaluation
python train.py --data_dir data --dataset agedb-dir --reweight sqrt_inv --loss l1 --epoch 1 --evaluate
```