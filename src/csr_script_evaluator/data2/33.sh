#!/bin/bash

# Training
python tools/train.py configs/ssdnerf_car.py --gpu-ids 0 --no-validate --max-epochs 1

# Inference / Demonstration
python tools/test.py configs/ssdnerf_car.py work_dirs/ssdnerf_car/latest.pth --gpu-ids 0 --save-results

# Testing / Evaluation
python tools/test.py configs/ssdnerf_car.py work_dirs/ssdnerf_car/latest.pth --gpu-ids 0 --eval mse,psnr,ssim,lpips
```