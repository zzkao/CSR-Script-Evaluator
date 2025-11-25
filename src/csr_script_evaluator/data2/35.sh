#!/bin/bash

# Training
python train.py --config configs/potsdam.yml --num_epochs 1 --batch_size 4

# Inference / Demonstration
python inference.py --config configs/potsdam.yml --checkpoint checkpoints/model.pth --input_image ./data/Potsdam/processed/test/image_001.png --output_dir ./results

# Testing / Evaluation
python eval.py --config configs/potsdam.yml --checkpoint checkpoints/model.pth
```