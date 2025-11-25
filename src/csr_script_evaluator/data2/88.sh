#!/bin/bash

# Training
python run.py --task_name SST-2 --data_dir data/k-shot/SST-2/16-13 --overwrite_output_dir --do_train --do_eval --do_predict --model_name_or_path roberta-base --few_shot_type prompt-demo --num_k 16 --max_steps 1000 --eval_steps 100 --per_device_train_batch_size 2 --learning_rate 1e-5 --num_train_epochs 1 --output_dir result/tmp --seed 13 --template "*cls**sent_0*_It_was*mask*.*sep+*" --mapping "{'0':'terrible','1':'great'}"

# Inference / Demonstration
python run.py --task_name SST-2 --data_dir data/k-shot/SST-2/16-13 --do_predict --model_name_or_path result/tmp --few_shot_type prompt-demo --num_k 16 --per_device_eval_batch_size 2 --output_dir result/tmp --template "*cls**sent_0*_It_was*mask*.*sep+*" --mapping "{'0':'terrible','1':'great'}"

# Testing / Evaluation
python run.py --task_name SST-2 --data_dir data/k-shot/SST-2/16-13 --do_eval --model_name_or_path result/tmp --few_shot_type prompt-demo --num_k 16 --per_device_eval_batch_size 2 --output_dir result/tmp --template "*cls**sent_0*_It_was*mask*.*sep+*" --mapping "{'0':'terrible','1':'great'}"
```