#!/bin/bash

# Training
# No training commands in README - this is an agent framework, not a model training repo

# Inference / Demonstration
export OPENAI_API_KEY='sk-XXX'
python run.py --model_name gpt4 --data_path https://github.com/pvlib/pvlib-python/issues/1603 --config_file config/default_from_url.yaml
python run.py --model_name gpt4 --data_path /path/to/data.jsonl --config_file ./config/default.yaml --per_instance_cost_limit 2.00
sweagent run --model_name gpt4 --data_path https://github.com/pvlib/pvlib-python/issues/1603 --config_file config/default_from_url.yaml

# Testing / Evaluation
# No explicit testing commands in README
```