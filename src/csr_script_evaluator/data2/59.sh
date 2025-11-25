#!/bin/bash

# Training
python main.py --exp_name curvenet_cls --num_points 1024 --k 20 --use_sgd True --scheduler cos --epochs 1

# Inference / Demonstration

# Testing / Evaluation
python main.py --exp_name curvenet_eval --num_points 1024 --k 20 --use_sgd True --eval True --model_path outputs/modelnet40/curvenet_cls/models/model.t7
```