#!/bin/bash

# Training
python3 training_and_testing/TYY_generators.py --batch_size 32 --nb_epochs 1 --db_name 300W_LP --image_size 64 --model_type 1

# Inference / Demonstration
python3 demo/demo_FSANET.py

# Testing / Evaluation
python3 training_and_testing/TYY_test_fsa.py
```