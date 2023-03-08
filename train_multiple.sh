#!/bin/bash

LOG_DIR="/home/neil/repos/thesis/logs/"
DEBUG_DIR="/home/neil/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/neil/miniconda3/envs/thesis/bin/python src/shared_queues/train_multiple.py --seed 1234 \
    --arch resnet18 resnet18 --epochs 3 --pretrained true false \
    --num-processes 2 --batch-size 70 --training-workers 4 --validation-workers 4 \
    --log_path /home/neil/repos/thesis/logs/queues
    #--debug_data_dir "${DEBUG_DIR}train_queues_debug"