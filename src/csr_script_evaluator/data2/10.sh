#!/bin/bash

# Training
cd EPro-PnP-6DoF
python train.py --cfg config/lm_obj01_train.yaml --obj_name obj_01 --epochs 1 --batch_size 4

# Inference / Demonstration
python demo.py --cfg config/lm_obj01_demo.yaml --obj_name obj_01 --ckpt_file checkpoints/lm_obj01.pth

# Testing / Evaluation
python test.py --cfg config/lm_obj01_test.yaml --obj_name obj_01 --ckpt_file checkpoints/lm_obj01.pth
```