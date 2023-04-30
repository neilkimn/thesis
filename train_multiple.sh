#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_imagenet/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_multiple_imagenet.py --seed 1234 \
    --arch resnet18 resnet18 --epochs 2 --pretrained true true \
    --num-processes 2 --batch-size 128 --training-workers 8 --validation-workers 2 \
    --log_dir "${LOG_DIR}queues" --record_first_batch_time
    #--debug_data_dir "${DEBUG_DIR}train_queues_debug"