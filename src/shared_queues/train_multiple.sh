#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs/"
DEBUG_DIR="/home/neni/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/neni/.venv/thesis/bin/python train_multiple.py --seed 1234 \
    --arch resnet34 resnet18 --epochs 1 --pretrained true true \
    --num-processes 2 --batch-size 80 --num-workers 4 \
    --debug_data_dir "${DEBUG_DIR}train_queues_debug"
    #--log_path /home/neni/repos/thesis/logs/queues