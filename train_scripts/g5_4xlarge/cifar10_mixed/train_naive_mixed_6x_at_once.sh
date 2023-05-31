#!/bin/bash

LOG_DIR="/home/ubuntu/repos/thesis/logs_g5_mixed/single_runs"
DEBUG_DIR="/home/ubuntu/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL="resnet18"
BATCH_SIZE=128
DATASET="cifar10"
MODEL_NAME="${MODEL}_mixed_bs_${BATCH_SIZE}"
EPOCHS=11

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet34" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet34" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet50" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/ubuntu/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet50" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 16 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

    #--debug_data_dir "${DEBUG_DIR}train_single_debug" &

training_main_proc=$!

echo "Starting training process with PID $training_main_proc"

mpstat 1 -P 0-15 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_cpu.out &
trace_cpu_pid=$!

nvidia-smi pmon -s um -o DT -f ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_gpu.out &
trace_gpu_pid=$!

iostat 1 -m -t nvme0n1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_io.out &
trace_io_pid=$!

dcgmi dmon -i 0 -e 200,201,203,204,210,211,1002,1003,1004,1005,1009,1010 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_dcgm.out &
dcgm_pid=$!

free -m -s 1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_free.out &
free_pid=$!

echo "Started mpstat (PID: $trace_cpu_pid), iostat (PID: $trace_io_pid), nvidia-smi (PID: $trace_gpu_pid), dcgmi (PID: $dcgm_pid) and free (PID: $free_pid)"

while kill -0 "$training_main_proc"; do
    sleep 5
done

sleep 20

kill $trace_cpu_pid
kill $trace_gpu_pid
kill $trace_io_pid
kill $dcgm_pid
kill $free_pid