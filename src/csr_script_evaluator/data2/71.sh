#!/bin/bash
# Training

# Inference / Demonstration
openai api completions.create -m text-davinci-003 -p "Say this is a test" -t 0 -M 7 --stream

# Testing / Evaluation

```