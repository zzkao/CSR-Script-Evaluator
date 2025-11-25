#!/bin/bash

# Training
python train.py --cfg configs/h36m/smoothnet.yaml --epoch 1 --batch_size 2

# Inference / Demonstration
python demo.py --cfg configs/demo/demo.yaml --video_path data/demo/example.mp4 --output_path results/demo/

# Testing / Evaluation
python test.py --cfg configs/h36m/smoothnet.yaml --checkpoint checkpoint/h36m/smoothnet.pth.tar
```