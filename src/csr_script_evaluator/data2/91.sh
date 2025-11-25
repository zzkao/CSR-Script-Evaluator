#!/bin/bash

# Training
accelerate launch --mixed_precision bf16 --num_processes 8 --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/supervised.py --dataset_name tatsu-lab/alpaca_farm --model_name_or_path EleutherAI/pythia-1.4b --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --learning_rate 2e-5 --output_dir output

# Inference / Demonstration
alpaca-farm evaluate --model_name_or_path output --evaluator_name gpt4

# Testing / Evaluation
alpaca-farm evaluate --model_name_or_path EleutherAI/pythia-1.4b --evaluator_name gpt4
```