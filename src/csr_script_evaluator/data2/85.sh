#!/bin/bash

# Training
python run.py --model ProteinKGAT --task ProteinTask --task_type multi_label_clf --data_path data/protein_dataset --pretrain_protein_model_path pretrained_model/OntoProtein_pretrain/OntoProtein.pth --epochs 1 --batch_size 32

# Inference / Demonstration
python predict.py --model ProteinKGAT --task ProteinTask --task_type multi_label_clf --data_path data/protein_dataset --ckpt_path ckpt/best_model.pth --input_seq MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL

# Testing / Evaluation
python evaluate.py --model ProteinKGAT --task ProteinTask --task_type multi_label_clf --data_path data/protein_dataset --ckpt_path ckpt/best_model.pth
```