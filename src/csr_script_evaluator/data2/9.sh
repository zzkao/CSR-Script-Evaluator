#!/bin/bash

# Training
# No training commands in README - this is an inference-only method

# Inference / Demonstration
python main.py --ni --config imagenet_256.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0 -i imagenet_256

# Testing / Evaluation
# Evaluation is part of the inference command above
```