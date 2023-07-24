#!/bin/bash

LOG_DIR="/home/ubuntu/repos/thesis/logs_g5/queues"
DEBUG_DIR="/home/ubuntu/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL=$1
BATCH_SIZE=4
DATASET="coco"
MODEL_NAME="${MODEL}_bs_${BATCH_SIZE}_cpu_${CPU}"
EPOCHS=2

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/opt/conda/envs/thesis/bin/python /home/ubuntu/repos/thesis/src/shared_queues/train_multiple_coco_tensorshare.py \
    --dataset coco --arch $MODEL $MODEL $MODEL $MODEL --epochs $EPOCHS --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --data-path /raid/datasets/$DATASET \
    --num-processes 4 --producer-per-worker --record_first_batch_time --batch-size $BATCH_SIZE \
    --log_dir "${LOG_DIR}/${DATASET}/${MODEL_NAME}" &
training_main_proc=$!

mpstat 1 -P 0-$CPU > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_cpu.out &
trace_cpu_pid=$!

nvidia-smi pmon -i 0 -s um -o DT -f ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_gpu.out &
trace_gpu_pid=$!

free -m -s 1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_free.out &
free_pid=$!

iostat 1 -m -t nvme0n1 > ${LOG_DIR}/${DATASET}/${MODEL_NAME}/pid_${training_main_proc}_io.out &
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
kill $free_pid