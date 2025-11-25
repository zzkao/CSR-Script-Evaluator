#!/bin/bash

# Training
python3 -m fastchat.train.train --model_name_or_path meta-llama/Llama-2-7b-hf --data_path ShareGPT_V3_unfiltered_cleaned_split.json --bf16 True --output_dir output_vicuna --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 10 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True

# Inference / Demonstration
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.gradio_web_server
python3 -m fastchat.serve.gradio_web_server_multi
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
python3 -m fastchat.serve.test_message --model-path lmsys/vicuna-7b-v1.5
python3 openai_api_demo.py

# Testing / Evaluation
python3 qa_browser.py --share
python3 -m fastchat.serve.monitor
python3 -m fastchat.eval.get_model_answer --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --num-gpus-total 1 --num-gpus-per-model 1 --max-new-tokens 1024
python3 -m fastchat.eval.eval_gpt_review
python3 -m fastchat.llm_judge.gen_model_answer --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
python3 -m fastchat.llm_judge.gen_judgment --model-list vicuna-7b-v1.5 --parallel 1
python3 -m fastchat.llm_judge.show_result
```