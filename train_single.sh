#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs/single_runs/"
DEBUG_DIR="/home/neni/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/neni/.venv/thesis/bin/python src/shared_queues/train_single.py --epochs 10 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 80 --training-workers 4 --validation-workers 4 \
    --log_path "${LOG_DIR}resnet18_bs80" &

/home/neni/.venv/thesis/bin/python src/shared_queues/train_single.py --epochs 10 \
    --seed 1234 --arch resnet34 --pretrained \
    --batch-size 80 --training-workers 4 --validation-workers 4 \
    --log_path "${LOG_DIR}resnet34_bs80"