#!/bin/bash

# Training
python train_model.py --epochs 1 --batch_size 1

# Inference / Demonstration
python sample_from_prior.py
python reconstruct.py

# Testing / Evaluation
python test_likelihoods.py
```