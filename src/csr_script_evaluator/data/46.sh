#!/bin/bash


# Data / Checkpoint / Weight Download (URL)
mkdir -p humansd_data/checkpoints
# Download required checkpoints from Google Drive and place them in the checkpoints directory:
# - higherhrnet_w48_humanart_512x512_udp.pth
# - v2-1_512-ema-pruned.ckpt
# - humansd-v1.ckpt
# - t2iadapter_openpose_sd14v1.pth
# - control_v11p_sd15_openpose.pth

# For dataset preparation (if needed)
mkdir -p humansd_data/datasets/Laion/Aesthetics_Human
python utils/download_data.py

# Training
# Standard training with heat-map-guided diffusion loss
python main.py --base configs/humansd/humansd-finetune.yaml -t --gpus 0,1 --name finetune_humansd
# Training without heat-map-guided diffusion loss (ablation)
python main.py --base configs/humansd/humansd-finetune-originalloss.yaml -t --gpus 0,1 --name finetune_humansd_original_loss

# Inference / Demonstration
# Basic demo with single pose
python scripts/pose2img.py --prompt "oil painting of girls dancing on the stage" --pose_file assets/pose/demo.npz
# Demo with comparison to ControlNet and T2I-Adapter
python scripts/pose2img.py --prompt "oil painting of girls dancing on the stage" --pose_file assets/pose/demo.npz --controlnet --t2i
# Gradio web interface
python scripts/gradio/pose2img.py
# Gradio with comparisons
python scripts/gradio/pose2img.py --controlnet --t2i

# Testing / Evaluation
python scripts/pose2img_metrics.py --outdir outputs/metrics --config utils/metrics/metrics.yaml --ckpt humansd_data/checkpoints/humansd-v1.ckpt