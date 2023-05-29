#!/bin/bash

# Naive
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_3x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_3x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_3x_at_once.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_3x_at_once.sh --use-dali

# Naive MPS
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_6x_at_once.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_naive_mixed_3x_at_once.sh --use-dali

# Shared
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_shared_mixed_3x_mps.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_shared_mixed_3x_mps.sh --use-dali

# Shared MPS
/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_shared_mixed_6x_mps.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/cifar10_mixed/train_shared_mixed_3x_mps.sh --use-dali

# Shared MPS weights
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/run_mps.sh /home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_2x_weights.sh
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/run_mps.sh /home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_3x_weights.sh

# Naive DALI
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_single_1x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_single_2x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_single_3x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_single_4x.sh --use-dali

# Shared DALI
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_1x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_2x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_3x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_4x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_5x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_6x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_7x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10/train_shared_8x.sh --use-dali