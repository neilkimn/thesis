#!/bin/bash
PATH+=:/home/neil/.venv/thesis/bin

if [ $# -eq 0 ]; then
    echo "Error: No first argument provided."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# set GPU to exclusive mode
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# start MPS control daemon
sudo nvidia-cuda-mps-control -d

# run some stuff ...
#./train_multiple.sh
LOG_DIR="/home/kafka/repos/thesis/logs/"
DEBUG_DIR="/home/kafka/repos/thesis/debug_data/"
CUDA_VISIBLE_DEVICES=0

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet18 --pretrained \
    --batch-size 50 --training-workers 4 --validation-workers 4 & #\
    #--log_path "${LOG_DIR}train_single_resnet18" &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
/home/kafka/miniconda3/envs/thesis/bin/python src/shared_queues/train_single.py --epochs 2 \
    --seed 1234 --arch resnet34 --pretrained \
    --batch-size 50 --training-workers 4 --validation-workers 4 #\
    #--log_path "${LOG_DIR}train_single_resnet34"

# shut down MPS control daemon
sudo echo quit | sudo nvidia-cuda-mps-control

# set GPU back to graphics mode
if [ "$1" = "gpu" ]; then
    sudo nvidia-smi -i 0 -c 0
    echo "Finished and set GPU to graphics mode."
elif [ "$1" = "dgx" ]; then
    echo "Finished."
fi

exit 0