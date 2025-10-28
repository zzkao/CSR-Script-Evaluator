#!/bin/bash

# Training
# (No explicit training command in README — zero-shot usage only.)

# Inference / Demonstration
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg sr_averagepooling --deg_scale 4.0 --sigma_y 0 -i demo

# Testing / Evaluation
sh evaluation.sh