#!/bin/bash

# Training
python gdtuo.py --optimizer_steps 200 --optimizer_lr 0.0001 --n_epochs 1 --training_steps 100

# Inference / Demonstration
python gdtuo.py

# Testing / Evaluation
# No explicit testing commands in README
```