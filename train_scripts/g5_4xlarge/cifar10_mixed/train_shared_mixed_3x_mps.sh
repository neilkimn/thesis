#!/bin/bash

LOG_DIR="/home/ubuntu/repos/thesis/logs_desktop_mixed/queues"
DEBUG_DIR="/home/ubuntu/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL="resnet18"
BATCH_SIZE=128
DATASET="cifar10"
MODEL_NAME="${MODEL}_bs_${BATCH_SIZE}"
EPOCHS=11

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_multiple.py \
    --arch resnet18 resnet34 resnet50 --epochs $EPOCHS --pretrained true true true --dataset $DATASET \
    --num-processes 3 --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_dir "${LOG_DIR}/${DATASET}/${MODEL_NAME}" --record_first_batch_time $1 & 
    #--debug_data_dir "${DEBUG_DIR}train_queues_debug" 

training_main_proc=$!

echo "Starting training process with PID $training_main_proc"

mpstat 1 -P 0-15 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_cpu.out &
trace_cpu_pid=$!

nvidia-smi pmon -s um -o DT -f ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_gpu.out &
trace_gpu_pid=$!

iostat 1 -m -t nvme0n1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_io.out &
trace_io_pid=$!

echo "Started mpstat (PID: $trace_cpu_pid), iostat (PID: $trace_io_pid) and nvidia-smi (PID: $trace_gpu_pid)"

while kill -0 "$training_main_proc"; do
    sleep 5
done

sleep 20

kill $trace_cpu_pid
kill $trace_gpu_pid
kill $trace_io_pid