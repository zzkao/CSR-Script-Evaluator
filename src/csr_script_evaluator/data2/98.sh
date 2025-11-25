#!/bin/bash

# Training
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm ERM --dataset ColoredMNIST --test_env 0 --output_dir ./output --steps 1

# Inference / Demonstration

# Testing / Evaluation
python -m domainbed.scripts.collect_results --input_dir ./output
```