#!/bin/bash

LOG_DIR="/home/neil/repos/thesis/logs/"
DEBUG_DIR="/home/neil/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/neil/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 100 --num-workers 4 &

/home/neil/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 100 --num-workers 4