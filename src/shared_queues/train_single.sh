#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs/"
DEBUG_DIR="/home/neni/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/neni/.venv/thesis/bin/python train_single.py --epochs 2 --seed 1234 --arch resnet18 --pretrained \
    --batch-size 80 --num-workers 4 &
    #--log_path "${LOG_DIR}train_single_1_debug" 
    #--debug_data_dir "${DEBUG_DIR}train_single_1_debug" &

/home/neni/.venv/thesis/bin/python train_single.py --epochs 2 --seed 1234 --arch resnet34 \
    --batch-size 80 --num-workers 4
    #--log_path "${LOG_DIR}train_single_2_debug" 
    #--debug_data_dir "${DEBUG_DIR}train_single_2_debug"