#!/bin/bash

# Training
python train.py --config configs/train_era5.yaml --trainer.max_epochs 1 --data.batch_size 1 --model.in_channels 73 --model.out_channels 73

# Inference / Demonstration
python scripts/sample.py --config configs/sample_era5.yaml --checkpoint checkpoints/dyffusion_era5.ckpt --num_samples 1

# Testing / Evaluation
python scripts/evaluate.py --config configs/eval_era5.yaml --checkpoint checkpoints/dyffusion_era5.ckpt --num_samples 1
```