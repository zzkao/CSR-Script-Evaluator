#!/bin/bash

# Training
bash scripts/v1_5/pretrain.sh
bash scripts/v1_5/finetune.sh

# Inference / Demonstration
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b
python -m llava.serve.cli --model-path liuhaotian/llava-v1.5-7b --image-file "https://llava-vl.github.io/static/images/view.jpg"

# Testing / Evaluation
bash scripts/v1_5/eval/llavabench.sh
bash scripts/v1_5/eval/mmbench.sh
bash scripts/v1_5/eval/pope.sh
bash scripts/v1_5/eval/seed.sh
bash scripts/v1_5/eval/sqa.sh
bash scripts/v1_5/eval/vizwiz.sh
bash scripts/v1_5/eval/vqav2.sh
bash scripts/v1_5/eval/gqa.sh
bash scripts/v1_5/eval/textvqa.sh
bash scripts/v1_5/eval/mme.sh
bash scripts/v1_5/eval/mmvet.sh
```