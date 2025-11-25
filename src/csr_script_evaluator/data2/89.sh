#!/bin/bash

# Training
python run_entity.py --do_train --do_eval --learning_rate=1e-5 --task_learning_rate=5e-4 --train_batch_size=8 --context_window=0 --task=scierc --data_dir=scierc_data/processed_data/json --model=allenai/scibert_scivocab_uncased --output_dir=scierc_models/ent-scib-ctx0 --num_train_epochs=1

# Inference / Demonstration
python run_entity.py --do_eval --eval_test --context_window=0 --task=scierc --data_dir=scierc_data/processed_data/json --model=scierc_models/ent-scib-ctx0 --output_dir=scierc_models/ent-scib-ctx0

# Testing / Evaluation
python run_relation.py --task=scierc --do_eval --eval_test --model=scierc_models/rel-scib-ctx0 --output_dir=scierc_models/rel-scib-ctx0
python run_eval.py --prediction_file=scierc_models/ent-scib-ctx0/predictions.json --task=scierc
python run_eval.py --prediction_file=scierc_models/rel-scib-ctx0/predictions.json --task=scierc
```