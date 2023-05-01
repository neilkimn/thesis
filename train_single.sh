#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_bpf/single_runs/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 1 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 50 --training-workers 1 --validation-workers 1 \
    --log_path "${LOG_DIR}resnet18_bs50" &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 1 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 50 --training-workers 1 --validation-workers 1 \
    --log_path "${LOG_DIR}resnet18_bs50" &

sudo bpftrace bpf_traces/bpf_read.bt -o logs_bpf/bpf_traces/trace_read_single.out &

#/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 10 \
#    --seed 1234 --arch resnet34 --pretrained \
#    --batch-size 80 --training-workers 4 --validation-workers 4 \
#    --log_path "${LOG_DIR}resnet34_bs80"