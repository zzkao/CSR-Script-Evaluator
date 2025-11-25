#!/bin/bash

# Training
python train_generation.py --category airplane --epochs 1

# Inference / Demonstration
python sample.py --ckpt pretrained/GEN_airplane.pt --categories airplane --batch_size 4

# Testing / Evaluation
python test_generation.py --ckpt pretrained/GEN_airplane.pt --categories airplane
```