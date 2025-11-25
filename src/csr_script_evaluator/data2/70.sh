#!/bin/bash

# Training
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-3.2-1B --output_dir ./output --batch_size_training 1 --num_epochs 1
python -m llama_recipes.finetuning --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-3.2-1B --fsdp_config.pure_bf16 --batch_size_training 1 --num_epochs 1
torchrun --nnodes 1 --nproc_per_node 1 examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-3.2-1B --batch_size_training 1 --num_epochs 1

# Inference / Demonstration
python -m llama_recipes.inference.chat_completion --model_name meta-llama/Llama-3.2-1B --prompt_file ./recipes/quickstart/chat_completion/prompt.txt --quantization
python -m llama_recipes.inference.chat_completion --model_name meta-llama/Llama-3.2-1B --prompt_file ./recipes/quickstart/chat_completion/prompt.txt --quantization --use_fast_kernels
python examples/inference.py --model_name meta-llama/Llama-3.2-1B --peft_model ./output --prompt_file ./recipes/quickstart/chat_completion/prompt.txt --quantization

# Testing / Evaluation
pytest tests/test_finetuning.py
python -m llama_recipes.inference.chat_completion --model_name meta-llama/Llama-3.2-1B --peft_model ./output --prompt_file ./recipes/quickstart/chat_completion/prompt.txt --quantization
```