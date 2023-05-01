#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_bpf/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_multiple.py --seed 1234 \
    --arch resnet18 resnet18 --epochs 1 --pretrained true true \
    --num-processes 2 --batch-size 50 --training-workers 1 --validation-workers 1 \
    --record_first_batch_time --log_dir "${LOG_DIR}queues" &
    #--debug_data_dir "${DEBUG_DIR}train_queues_debug"
python_pid=$!

echo "Python process started with PID $python_pid"

sudo bpftrace bpf_traces/bpf_read.bt -o logs_bpf/bpf_traces/trace_read_shared.out &