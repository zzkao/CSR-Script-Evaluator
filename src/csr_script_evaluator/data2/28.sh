#!/bin/bash

# Training
python run/pose2d/train.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml --gpu 0 --epoch 1

# Inference / Demonstration
python run/pose2d/valid.py --cfg experiments-local/mixed/resnet50/256_fusion.yaml --gpu 0
python run/pose3d/estimate.py --cfg experiments-local/shelf/prn64.yaml --gpu 0

# Testing / Evaluation
python run/pose3d/evaluate.py --cfg experiments-local/shelf/prn64.yaml --gpu 0
python run/pose3d/evaluate.py --cfg experiments-local/campus/prn64.yaml --gpu 0
```