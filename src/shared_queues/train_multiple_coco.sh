#!/bin/bash

LOG_DIR="/home/kafka/repos/thesis/logs_desktop/queues"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL="rcnn"
BATCH_SIZE=1
DATASET="coco"
MODEL_NAME="${MODEL}_bs_${BATCH_SIZE}"
EPOCHS=1

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/home/kafka/miniconda3/envs/thesis/bin/python /home/kafka/repos/thesis/src/shared_queues/train_multiple_coco.py \
    --epochs $EPOCHS --pretrained true true \
    --num-processes 2 --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_dir "${LOG_DIR}/${DATASET}/${MODEL_NAME}" --record_first_batch_time $1 \
    --model fasterrcnn_resnet50_fpn --config coco.yaml