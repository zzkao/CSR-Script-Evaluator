#!/bin/bash

# Training
python train.py --model_name_or_path bert-base-uncased --train_file data/wiki1m_for_simcse.txt --output_dir result/my-unsup-simcse-bert-base-uncased --num_train_epochs 1 --per_device_train_batch_size 64 --learning_rate 3e-5 --max_seq_length 32 --evaluation_strategy steps --metric_for_best_model stsb_spearman --load_best_model_at_end --eval_steps 125 --pooler_type cls --mlp_only_train --overwrite_output_dir --temp 0.05 --do_train --fp16

# Inference / Demonstration
python simcse_to_huggingface.py --path result/my-unsup-simcse-bert-base-uncased
python demo.py

# Testing / Evaluation
python evaluation.py --model_name_or_path result/my-unsup-simcse-bert-base-uncased --pooler cls_before_pooler --task_set sts --mode test
```