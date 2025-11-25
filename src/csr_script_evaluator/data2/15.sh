#!/bin/bash

# Training
python train.py --dataset cifar10 --model resnet18 --loss triplet --epochs 1 --batch_size 32

# Inference / Demonstration
python test.py --dataset cifar10 --model resnet18 --loss triplet --batch_size 32

# Testing / Evaluation
python test.py --dataset cifar10 --model resnet18 --loss triplet --batch_size 32
```