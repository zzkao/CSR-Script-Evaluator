#!/bin/bash

# Training
python train.py --config configs/nerf/lego.txt --expname lego_raydf --num_epochs 1

# Inference / Demonstration
python render.py --config configs/nerf/lego.txt --ckpt ckpts/lego/lego.tar --render_test

# Testing / Evaluation
python eval.py --config configs/nerf/lego.txt --ckpt ckpts/lego/lego.tar
```