#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_multiple_mps.py --seed 1234 \
    --arch resnet18 resnet34 --epochs 10 --pretrained true true \
    --num-processes 2 --batch-size 50 --training-workers 4 --validation-workers 4 \
    --log_dir /home/kafka/repos/thesis/logs/queues --record_first_batch_time
    #--debug_data_dir "${DEBUG_DIR}train_queues_debug"