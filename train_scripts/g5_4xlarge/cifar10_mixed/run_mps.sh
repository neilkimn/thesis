#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: No first argument provided."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0
CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1

# set GPU to exclusive mode
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# start MPS control daemon
sudo nvidia-cuda-mps-control -d

# run some stuff ...
bash $1 $2

# shut down MPS control daemon
sudo echo quit | sudo nvidia-cuda-mps-control

# set GPU back to graphics mode
sudo nvidia-smi -i 0 -c 0
echo "Finished and set GPU to graphics mode."

exit 0