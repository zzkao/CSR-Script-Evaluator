#!/bin/bash

# Training

# Inference / Demonstration
python app.py
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 1 --n_iter 1 --scale 7.5 --ddim_steps 50 --ckpt models/humansd-v1.ckpt --prompt "a photo of a person" --outdir outputs/txt2img-samples
python scripts/stable_img2img.py --ddim_eta 0.0 --n_samples 1 --n_iter 1 --scale 5.0 --strength 0.8 --ddim_steps 50 --ckpt models/humansd-v1.ckpt --init-img inputs/demo.jpg --prompt "a photo of a person" --outdir outputs/img2img-samples

# Testing / Evaluation

```