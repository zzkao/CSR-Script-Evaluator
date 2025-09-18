
#!/bin/bash
# Environment Setup / Requirement / Installation
pip install torch fairscale fire tiktoken==0.4.0 blobfile
pip install -e .
pip install huggingface-hub
chmod +x download.sh

# Data / Checkpoint / Weight Download (URL)
./download.sh
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct

# Training
# Note: No specific training commands found in this repository (inference-focused)

# Inference / Demonstration
torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 512 --max_batch_size 6
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir Meta-Llama-3-8B/ --tokenizer_path Meta-Llama-3-8B/tokenizer.model --max_seq_len 128 --max_batch_size 4

# Testing / Evaluation
# Note: No specific testing/evaluation commands found in README
