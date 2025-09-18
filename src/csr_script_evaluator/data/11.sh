#!/bin/bash
# Data / Checkpoint / Weight Download (URL)
curl -L -o models_and_data.zip "https://drive.google.com/uc?export=download&id=1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k"
unzip models_and_data.zip
cp -r shared_models/* output/
cp -r shared_data/* datasets/VOC2007/

# Training
python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1"
cp -r ./output/t1 ./output/t2
python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "./output/t2/model_final.pth"
cp -r ./output/t2 ./output/t2_ft
python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
cp -r ./output/t2_ft ./output/t3
python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3" MODEL.WEIGHTS "./output/t3/model_final.pth"
cp -r ./output/t3 ./output/t3_ft
python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_ft" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"

# Inference / Demonstration
python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"
python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"

# Testing / Evaluation
python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"
python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"
python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"
python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t4_final" MODEL.WEIGHTS "./output/t4_ft/model_final.pth"
