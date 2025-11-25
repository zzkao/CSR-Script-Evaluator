#!/bin/bash
# Training
python main.py --data ~/datasets/ILSVRC2012 --arch alexnet_lpf --lr 0.01 --batch-size 256 --epochs 1

# Inference / Demonstration
# No specific inference commands in README

# Testing / Evaluation
python main.py --data ~/datasets/ILSVRC2012 --arch alexnet_lpf --weights antialiased_alexnet.pth.tar --evaluate
```