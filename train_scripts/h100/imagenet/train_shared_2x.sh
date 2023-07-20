#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs_h100/queues"
DEBUG_DIR="/home/neni/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL=$1
BATCH_SIZE=128
DATA_DIR="/raid/datasets/"
#DATA_DIR="/home/neni/datasets/"
DATASET="imagenet"
MODEL_NAME="${MODEL}_bs_${BATCH_SIZE}"
EPOCHS=2

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/home/neni/.conda/envs/pt/bin/python /home/neni/repos/thesis/src/shared_queues/train_multiple_timm.py \
    --arch $MODEL $MODEL --epochs $EPOCHS --pretrained true true --lr 0.1 --data-dir $DATA_DIR$DATASET \
    --num-processes 2 --batch-size $BATCH_SIZE --training-workers 12 --validation-workers 8 --num-classes 1000 \
    --gpu-prefetch \
    --log_dir "${LOG_DIR}/${DATASET}/${MODEL_NAME}" --record_first_batch_time &
training_main_proc=$!

mpstat 1 -P 0-47 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_cpu.out &
trace_cpu_pid=$!

nvidia-smi pmon -i 0 -s um -o DT -f ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_gpu.out &
trace_gpu_pid=$!

iostat 1 -m -t nvme1n1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_io.out &
trace_io_pid=$!

dcgmi dmon -i 0 -e 200,201,203,204,210,211,1002,1003,1004,1005,1009,1010 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_dcgm.out &
dcgm_pid=$!

echo "Started mpstat (PID: $trace_cpu_pid), iostat (PID: $trace_io_pid), nvidia-smi (PID: $trace_gpu_pid) and dcgmi (PID: $dcgm_pid)"

while kill -0 "$training_main_proc"; do
    sleep 5
done

sleep 20

kill $trace_cpu_pid
kill $trace_gpu_pid
kill $trace_io_pid
kill $dcgm_pid