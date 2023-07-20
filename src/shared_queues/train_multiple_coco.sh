#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs_desktop/queues"
DEBUG_DIR="/home/neni/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

MODEL="fasterrcnn_resnet50_fpn"
BATCH_SIZE=1
DATASET="coco"
MODEL_NAME="${MODEL}_bs_${BATCH_SIZE}"
EPOCHS=1

sleep 1
if [[ ! -e ${LOG_DIR}/${DATASET}/${MODEL_NAME} ]]; then
    mkdir -p ${LOG_DIR}/${DATASET}/${MODEL_NAME}
fi

#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"

/home/neni/.conda/envs/pt/bin/python /home/neni/repos/thesis/src/shared_queues/train_multiple_coco_2.py \
    --epochs $EPOCHS --pretrained true \
    --num-processes 1 --batch-size $BATCH_SIZE --training-workers 8 --validation-workers 1 \
    --log_dir "${LOG_DIR}/${DATASET}/${MODEL_NAME}" --record_first_batch_time $1 \
    --model $MODEL --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --data-path /home/neni/datasets/coco_minitrain_25k