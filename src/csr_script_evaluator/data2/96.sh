#!/bin/bash
# Training
python run.py --model_name_or_path bert-base-uncased --data_dir ./data/semeval --output_dir ./output/semeval --do_train --do_eval --learning_rate 3e-5 --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --evaluation_strategy epoch --save_strategy epoch --seed 42

# Inference / Demonstration
python run.py --model_name_or_path bert-base-uncased --data_dir ./data/semeval --output_dir ./output/semeval --do_predict --per_device_eval_batch_size 16

# Testing / Evaluation
python run.py --model_name_or_path bert-base-uncased --data_dir ./data/semeval --output_dir ./output/semeval --do_eval --per_device_eval_batch_size 16
```