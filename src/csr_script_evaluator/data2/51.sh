#!/bin/bash

# Training
python train.py --dataset_dir ./fivek_dataset --num_epoch 1 --batch_size 1 --learning_rate 0.0001

# Inference / Demonstration
python test.py --input_img ./example_input.jpg --output_img ./example_output.jpg --model_path ./checkpoints/curl_model.pth

# Testing / Evaluation
python evaluate.py --dataset_dir ./fivek_dataset --model_path ./checkpoints/curl_model.pth
```