#!/bin/bash

# Training
python scripts/train_smpl.py --cfg configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml --batch-size 24 --epochs 1

# Inference / Demonstration
python scripts/demo_image.py --img-dir examples --out-dir res

# Testing / Evaluation
python scripts/validate_smpl_cam.py --cfg configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml --batch-size 24
```