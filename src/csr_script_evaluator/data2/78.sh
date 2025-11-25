#!/bin/bash

# Training
# No training commands in README

# Inference / Demonstration
export OPENAI_API_KEY=your_key_here
python run.py --task game24 --method_generate sample --method_evaluate value --method_select greedy --n_generate_sample 1 --n_evaluate_sample 1 --n_select_sample 1
python run.py --task text --method_generate sample --method_evaluate value --method_select greedy --n_generate_sample 1 --n_evaluate_sample 1 --n_select_sample 1
python run.py --task crosswords --method_generate sample --method_evaluate value --method_select greedy --n_generate_sample 1 --n_evaluate_sample 1 --n_select_sample 1

# Testing / Evaluation
# No explicit testing commands in README
```