#!/bin/bash

# Training
python train.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --do_train --do_eval --output_dir ./output --overwrite_output_dir --num_train_epochs 1 --learning_rate 5e-5 --block_size 128 --save_strategy epoch --evaluation_strategy epoch --logging_steps 100

# Inference / Demonstration

# Testing / Evaluation
python evaluate.py --model_name_or_path ./output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_eval_batch_size 1 --block_size 128
```