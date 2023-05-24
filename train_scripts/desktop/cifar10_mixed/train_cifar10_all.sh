#!/bin/bash

# Naive
/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_naive_mixed_3x.sh

# Naive MPS
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/run_mps.sh

# Shared
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_2x_mps.sh
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_3x_mps.sh

# Shared MPS
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/run_mps.sh /home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_2x_mps.sh
#/home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/run_mps.sh /home/kafka/repos/thesis/train_scripts/desktop/cifar10_mixed/train_shared_mixed_3x_mps.sh

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