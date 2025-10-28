#!/bin/bash

# Training
# No training commands found - 3DDFA_V2 is an inference-only system

# Inference / Demonstration
python demo.py -f examples/inputs/emma.jpg --onnx
python demo_video.py -f examples/inputs/videos/214.avi --onnx
python demo_video_smooth.py -f examples/inputs/videos/214.avi --onnx

# Testing / Evaluation
python latency.py
python latency.py --onnx
