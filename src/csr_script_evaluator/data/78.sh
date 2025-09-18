#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
# Note: This framework uses built-in tasks and data, no external downloads needed

# Training
# Note: Tree of Thought is an inference/reasoning framework, not a training system

# Inference / Demonstration
python run.py --task game24 --task_start_index 900 --task_end_index 910 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 910 --naive_run --prompt_sample standard --n_generate_sample 10 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 910 --naive_run --prompt_sample cot --n_generate_sample 10 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task text --task_start_index 0 --task_end_index 10 --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample 5 --n_evaluate_sample 5 --n_select_sample 1 --prompt_sample cot --temperature 1.0 --backend gpt-3.5-turbo
python run.py --task text --task_start_index 0 --task_end_index 10 --naive_run --prompt_sample standard --n_generate_sample 10 --temperature 1.0 --backend gpt-3.5-turbo
sh scripts/game24/standard_sampling.sh --backend gpt-3.5-turbo
sh scripts/game24/cot_sampling.sh --backend gpt-3.5-turbo
sh scripts/game24/bfs.sh --backend gpt-3.5-turbo
sh scripts/text/standard_sampling.sh --backend gpt-3.5-turbo
sh scripts/text/bfs.sh --backend gpt-3.5-turbo

# Testing / Evaluation
python run.py --task game24 --task_start_index 900 --task_end_index 1000 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task game24 --task_start_index 900 --task_end_index 1000 --naive_run --prompt_sample standard --n_generate_sample 100 --backend gpt-3.5-turbo --temperature 0.7
python run.py --task text --task_start_index 0 --task_end_index 100 --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample 5 --n_evaluate_sample 5 --n_select_sample 1 --prompt_sample cot --temperature 1.0 --backend gpt-3.5-turbo
python run.py --task crosswords --task_start_index 0 --task_end_index 10 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5 --backend gpt-3.5-turbo --temperature 0.7