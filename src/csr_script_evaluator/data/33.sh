#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
mkdir -p ./data/shapenet
mkdir -p ./data/abo
mkdir -p ./data/kitti/training
wget -O srn_cars.zip "https://drive.google.com/uc?export=download&id=1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR"
wget -O srn_chairs.zip "https://drive.google.com/uc?export=download&id=1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR"
wget -O abo_tables.zip "https://drive.google.com/uc?export=download&id=1lzw3uYbpuCxWBYYqYyL4ZEFomBOUN323"
unzip srn_cars.zip -d ./data/shapenet/
unzip srn_chairs.zip -d ./data/shapenet/
unzip abo_tables.zip -d ./data/abo/

# Training
python train.py configs/new_cfgs/ssdnerf_cars_uncond_16bit.py --gpu-ids 0
python train.py configs/new_cfgs/ssdnerf_cars_recons1v_16bit.py --gpu-ids 0
python train.py configs/paper_cfgs/ssdnerf_cars_uncond.py --gpu-ids 0 1
python train.py configs/paper_cfgs/ssdnerf_chairs_recons1v.py --gpu-ids 0 1
python train.py configs/paper_cfgs/ssdnerf_abotables_uncond.py --gpu-ids 0 1
python train.py configs/new_cfgs/stage1_cars_recons16v_16bit.py --gpu-ids 0 1
python train.py configs/paper_cfgs/stage1_cars_recons16v.py --gpu-ids 0 1
python train.py configs/paper_cfgs/ssdnerf_cars3v_uncond_1m.py --gpu-ids 0 1
python train.py configs/paper_cfgs/ssdnerf_cars3v_uncond_2m.py --gpu-ids 0 1 --resume-from work_dirs/ssdnerf_cars3v_uncond_1m/latest.pth

# Inference / Demonstration
python demo/ssdnerf_gui.py configs/paper_cfgs/ssdnerf_cars_uncond.py work_dirs/ssdnerf_cars_uncond/latest.pth --fp16
python demo/ssdnerf_gui.py configs/new_cfgs/ssdnerf_cars_uncond_16bit.py work_dirs/ssdnerf_cars_uncond_16bit/latest.pth --fp16
python test.py configs/paper_cfgs/ssdnerf_cars_uncond.py work_dirs/ssdnerf_cars_uncond/latest.pth --gpu-ids 0
python test.py configs/paper_cfgs/ssdnerf_chairs_recons1v.py work_dirs/ssdnerf_chairs_recons1v/latest.pth --gpu-ids 0
python test.py configs/new_cfgs/ssdnerf_cars_recons1v_16bit.py work_dirs/ssdnerf_cars_recons1v_16bit/latest.pth --gpu-ids 0 1

# Testing / Evaluation
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/paper_cfgs/ssdnerf_cars_uncond.py
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/paper_cfgs/ssdnerf_chairs_recons1v.py
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/paper_cfgs/ssdnerf_abotables_uncond.py
python tools/kitti_preproc.py
python test.py configs/paper_cfgs/ssdnerf_cars_recons1v.py work_dirs/ssdnerf_cars_recons1v/latest.pth --gpu-ids 0 1
python test.py configs/supp_cfgs/ssdnerf_cars_reconskitti.py work_dirs/ssdnerf_cars_recons1v/latest.pth --gpu-ids 0
python test.py configs/paper_cfgs/multiview_recons/ssdnerf_cars_recons4v.py work_dirs/ssdnerf_cars_recons1v/latest.pth --gpu-ids 0