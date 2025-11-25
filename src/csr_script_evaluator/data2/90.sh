#!/bin/bash

# Training
# No training commands in this repository - it's an evaluation framework

# Inference / Demonstration
alpaca_eval --model_outputs 'example/outputs.json'
alpaca_eval --model_outputs 'example/outputs.json' --annotators_config 'alpaca_eval_gpt4'
alpaca_eval --model_outputs 'example/outputs.json' --reference_outputs 'example/reference_outputs.json'
alpaca_eval make_leaderboard --leaderboard_mode_to_print minimal

# Testing / Evaluation
alpaca_eval evaluate --model_outputs 'example/outputs.json'
alpaca_eval evaluate_from_model --model_configs 'oasst_pythia_12b' --max_instances 1
alpaca_eval analyze_evaluators --annotators_config 'alpaca_eval_gpt4_turbo_fn'
```