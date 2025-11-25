#!/bin/bash

# Training
python train.py --config configs/garment.yaml --gpu 0 --nepoch 1

# Inference / Demonstration
python test.py --config configs/garment.yaml --gpu 0

# Testing / Evaluation
python test.py --config configs/garment.yaml --gpu 0
```