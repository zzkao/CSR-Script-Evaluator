#!/bin/bash

# Training
python -m torch_mimicry.training.train_gan --dataset cifar10 --arch sngan_32 --max_steps 100000

# Inference / Demonstration
# No explicit inference commands in README

# Testing / Evaluation
python -m torch_mimicry.metrics.compute_fid --netG_ckpt_file /path/to/netG_ckpt.pth --dataset cifar10 --num_samples 50000
```