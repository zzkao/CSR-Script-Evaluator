#!/bin/bash

# Training
python scripts/train_diffusion.py configs/qm9_default.yml

# Inference / Demonstration
python scripts/sample_diffusion.py configs/qm9_default.yml --ckpt_path ./logs/qm9_default/checkpoints/best_model.pt --num_samples 100

# Testing / Evaluation
python scripts/evaluate.py configs/qm9_default.yml --ckpt_path ./logs/qm9_default/checkpoints/best_model.pt
```