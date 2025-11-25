#!/bin/bash

# Training
python train.py --dataset cifar10 --arch vgg16_bn --epochs 1 --batch-size 128

# Inference / Demonstration
python attack.py --dataset cifar10 --arch vgg16_bn --attack pgd --epsilon 0.031 --attack-iter 10

# Testing / Evaluation
python test.py --dataset cifar10 --arch vgg16_bn
```