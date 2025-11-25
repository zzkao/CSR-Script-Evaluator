#!/bin/bash

# Training
python train.py --config configs/nyu_rgb.yaml --epoch 1 --batch_size 2

# Inference / Demonstration
python demo.py --config configs/nyu_rgb.yaml --checkpoint checkpoints/best_model.pth --input demo/rgb.png

# Testing / Evaluation
python test.py --config configs/nyu_rgb.yaml --checkpoint checkpoints/best_model.pth
```