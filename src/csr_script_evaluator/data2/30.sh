#!/bin/bash

# Training
python -m graphsaint.tensorflow_version.train --data_prefix ./data/ppi --train_config ./train_config/ppi.yml --gpu 0

# Inference / Demonstration
# No explicit inference commands found in README

# Testing / Evaluation
python -m graphsaint.tensorflow_version.train --data_prefix ./data/ppi --train_config ./train_config/ppi.yml --gpu 0 --eval_train_every 5
```