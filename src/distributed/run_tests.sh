#!/bin/bash

LOG_DIR="/home/neni/repos/thesis/logs/"

CUDA_VISIBLE_DEVICES=0,1,2

sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
python 9_distributed.py -a resnet18 --epochs 5 -b 450 -j 16 --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j16_dev012" /home/neni/repos/data_loading_ads/data
sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
python 9_distributed.py -a resnet18 --epochs 5 -b 450 -j 16 --dummy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j16_dev012_dummy" /home/neni/repos/data_loading_ads/data
sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
python 9_distributed.py -a resnet18 --epochs 5 -b 450 -j 16 --dummy --no-memcpy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j16_dev012_dummy_no_memcpy" /home/neni/repos/data_loading_ads/data

#CUDA_VISIBLE_DEVICES=0,1
#
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 300 -j 8 --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://#127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev01" /home/neni/repos/data_loading_ads/data
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 300 -j 8 --dummy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url #'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev01_dummy" /home/neni/repos/data_loading_ads/data
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 300 -j 8 --dummy --no-memcpy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev01_dummy_no_memcpy" /home/neni/repos/data_loading_ads/data

#CUDA_VISIBLE_DEVICES=0
#
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 150 -j 8 --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://#127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev0" /home/neni/repos/data_loading_ads/data
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 150 -j 8 --dummy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url #'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev0_dummy" /home/neni/repos/data_loading_ads/data
#sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"
#python 9_distributed.py -a resnet18 --epochs 5 -b 150 -j 8 --dummy --no-memcpy --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:12355' --name "${LOG_DIR}resnet18_b150_j8_dev0_dummy_no_memcpy" /home/neni/repos/data_loading_ads/data