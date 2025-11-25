#!/bin/bash

# Training
python train.py --cfg configs/res_regression_r50.yaml --data-path data/coco --epochs 1 --batch-size 8

# Inference / Demonstration
python demo.py --cfg configs/res_regression_r50.yaml --weights res_regression_weights/res_regression_r50.pth --img demo/000000000139.jpg

# Testing / Evaluation
python test.py --cfg configs/res_regression_r50.yaml --weights res_regression_weights/res_regression_r50.pth --data-path data/coco
```