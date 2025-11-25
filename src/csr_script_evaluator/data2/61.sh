#!/bin/bash

# Training
python train.py --model_name_or_path facebook/opt-125m --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir ./output --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --num_train_epochs 1 --learning_rate 2e-5 --logging_steps 100 --save_steps 1000 --evaluation_strategy steps --eval_steps 1000

# Inference / Demonstration
python inference.py --model_name_or_path ./output --input_text "The quick brown fox"

# Testing / Evaluation
python evaluate.py --model_name_or_path ./output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_eval_batch_size 1
```