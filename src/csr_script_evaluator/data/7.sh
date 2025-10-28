#!/bin/bash

# Training
python run.py --config experiments/train_celeba.yml --gpu 0 --num_workers 4

# Inference / Demonstration
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth --gpu

# Testing / Evaluation
