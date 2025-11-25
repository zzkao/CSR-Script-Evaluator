#!/bin/bash

# Training
python main.py --scale 2 --patch_size 32 --batch_size 16 --max_steps 1 --decay 200 --model MSRN --ckpt_name MSRN_x2 --ckpt_dir ./checkpoint/MSRN_x2 --scale 2 --num_features 64 --num_blocks 8 --res_scale 1

# Inference / Demonstration
python main.py --scale 2 --ckpt_name MSRN_x2 --test_only --ckpt_dir ./checkpoint/MSRN_x2

# Testing / Evaluation
python main.py --scale 2 --ckpt_name MSRN_x2 --test_only --ckpt_dir ./checkpoint/MSRN_x2 --self_ensemble
```