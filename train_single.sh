#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_imagenet/single_runs/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 11 \
#    --seed 1234 --arch resnet18 --pretrained \
#    --batch-size 50 --training-workers 8 --validation-workers 2 \
#    --log_path "${LOG_DIR}resnet18_bs50"

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single_imagenet.py \
    --epochs 2 --seed 1234 --arch resnet18 --pretrained \
    --batch-size 128 --training-workers 8 --validation-workers 2 \
    --log_path "${LOG_DIR}resnet18_bs128" &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single_imagenet.py \
    --epochs 2 --seed 1234 --arch resnet18 --pretrained \
    --batch-size 128 --training-workers 8 --validation-workers 2 \
    --log_path "${LOG_DIR}resnet18_bs128"