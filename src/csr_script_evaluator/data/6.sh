#!/bin/bash

# Inference / Demonstration
python - <<'EOF'
from mmpretrain import inference_model
predict = inference_model('resnet50_spark-pre_300e_in1k','demo/bird.JPEG')
print(predict['pred_class'], predict['pred_score'])
EOF  # using the demo snippet from docs :contentReference[oaicite:3]{index=3}

# Testing / Evaluation
python tools/test.py configs/spark/benchmarks/resnet50_8xb256-coslr-300e_in1k.py resnet50_8xb256-coslr-300e_in1k_20230612-f86aab51.pth
