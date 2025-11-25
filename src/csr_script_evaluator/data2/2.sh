#!/bin/bash

# Training
# No training commands provided in README

# Inference / Demonstration
python3 demo.py -f examples/inputs/emma.jpg
python3 demo.py -f examples/inputs/trump_hillary.jpg
python3 demo.py -f examples/inputs/ad.jpg
python3 demo.py -f examples/inputs/AD_4.jpg
python3 demo.py -f examples/inputs/AF_1.jpg
python3 demo.py -m gpu -f examples/inputs/emma.jpg
python3 demo.py -m onnx -f examples/inputs/emma.jpg
sh ./run_test.sh

# Testing / Evaluation
python3 benchmark.py
```