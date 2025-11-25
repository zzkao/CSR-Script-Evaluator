#!/bin/bash

# Training
python train.py -s data/tandt/train -m output/train --iterations 1000

# Inference / Demonstration
python render.py -m output/train

# Testing / Evaluation
python metrics.py -m output/train
```