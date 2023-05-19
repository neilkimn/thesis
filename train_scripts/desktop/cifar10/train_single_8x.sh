#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_desktop/single_runs"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
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

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py \
    --log-interval 10 --epochs $EPOCHS --arch "resnet18" --pretrained --dataset $DATASET \
    --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_path "${LOG_DIR}/${DATASET}/${MODEL_NAME}" $1 &

    #--debug_data_dir "${DEBUG_DIR}train_single_debug" &

training_main_proc=$!

echo "Starting training process with PID $training_main_proc"

sleep 1

mpstat 1 -P 0-5 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_cpu.out &
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