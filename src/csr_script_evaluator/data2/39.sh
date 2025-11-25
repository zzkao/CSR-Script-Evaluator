#!/bin/bash

# Training
python scripts/train.py --config config/config.py --epochs 1

# Inference / Demonstration
python scripts/inference.py --config config/config.py --checkpoint output/checkpoint.pth --input demo/input --output demo/output

# Testing / Evaluation
python scripts/evaluate.py --config config/config.py --checkpoint output/checkpoint.pth
```