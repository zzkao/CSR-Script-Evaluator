#!/bin/bash

# Training
python train.py --model_name facebook/opt-1.3b --task_name SST2 --trainer zo --load_best_model_at_end --zo_eps 1e-3 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --num_train_epochs 1 --learning_rate 1e-6 --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end --output_dir output

# Inference / Demonstration
# No explicit inference commands in README

# Testing / Evaluation
# No explicit testing commands in README
```