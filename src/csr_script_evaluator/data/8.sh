#!/bin/bash

# Training
# (Assuming script supports config, exp_id; using default “cpu” and epoch=1 as requested)
python train.py --cfg configs/hybrik/h36m_ResNet50.yaml --exp_id hybrik_test --device cpu --epochs 1

# Inference / Demonstration
python demo.py --cfg configs/hybrik/h36m_ResNet50.yaml --checkpoint checkpoints/hybrik_test/model_best.pth.tar --image examples/image.jpg --out_dir results/demo

# Testing / Evaluation
python test.py --cfg configs/hybrik/h36m_ResNet50.yaml --checkpoint checkpoints/hybrik_test/model_best.pth.tar --dataset 3DPW --device cpu
