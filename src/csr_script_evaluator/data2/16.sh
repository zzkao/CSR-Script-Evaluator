#!/bin/bash
# Training
python train.py --dataset cifar10 --num-labeled 1500 --num-classes 10 --arch wideresnet --batch-size 64 --epochs 1 --lr 0.03 --seed 5 --out results/cifar10@1500

# Inference / Demonstration

# Testing / Evaluation
```