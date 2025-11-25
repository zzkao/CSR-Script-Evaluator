#!/bin/bash

# Training
python examples/classification/run_classification.py --model_name_or_path bert-base-cased --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --output_dir /tmp/sst2 --overwrite_output_dir --disable_tqdm True --per_sample_max_grad_norm 1.0 --target_epsilon 3 --seed 0

# Inference / Demonstration
# No explicit inference commands in README

# Testing / Evaluation
python -m unittest discover -s tests
```