#!/bin/bash

# Training
# (No explicit full training command given for custom dataset in README — skipping)

# Inference / Demonstration
python example_usage.py
python example_usage2.py

# Testing / Evaluation
python main.py --dataset imagenet --model resnet50 --pretrained True --filter_size 4
