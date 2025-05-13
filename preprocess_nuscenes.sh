#!/bin/bash

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=6 python datasets/preprocess.py \
    --data_root data/nuscenes/raw/nuscenes/ \
    --target_dir data/nuscenes/processed\
    --dataset nuscenes \
    --split v1.0-trainval \
    --start_idx 832 \
    --num_scenes 10 \
    --interpolate_N 4 \
    --workers 32 \
    --process_keys images lidar calib dynamic_masks objects