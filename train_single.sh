#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 50 --training-workers 4 --validation-workers 4 \
    --log_path "${LOG_DIR}train_single_resnet18"

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 50 --training-workers 4 --validation-workers 4 \
    --log_path "${LOG_DIR}train_single_resnet18"