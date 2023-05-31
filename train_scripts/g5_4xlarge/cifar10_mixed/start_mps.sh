#!/bin/bash

sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
export CUDA_MPS_LOG_DIRECTORY=/home/ubuntu/mps_log$1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1

echo "starting MPS server with active thread percentage:" $1
nvidia-cuda-mps-control -d