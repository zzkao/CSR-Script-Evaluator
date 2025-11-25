#!/bin/bash

# Training
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/checkpoint.pt

# Inference / Demonstration
python -u sample.py --config-path=/path/to/checkpoint/folder --n_samples=1 ++mode=sample ++n_samples=512 ++model.eval_batch_size=32 ++samples_dir=samples/

# Testing / Evaluation
# No specific testing commands found in README
```