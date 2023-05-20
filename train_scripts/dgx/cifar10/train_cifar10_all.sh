#!/bin/bash

# Naive
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_1x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_2x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_3x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_4x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_5x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_6x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_7x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_8x.sh

# Naive MPS
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_1x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_2x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_3x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_4x.sh

# Shared
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_1x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_2x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_3x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_4x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_5x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_6x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_7x.sh
/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_8x.sh

# Shared MPS
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_1x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_2x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_3x.sh
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/run_mps.sh /home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_4x.sh

# Naive DALI
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_1x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_2x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_3x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_single_4x.sh --use-dali

# Shared DALI
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_1x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_2x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_3x.sh --use-dali
#/home/neni/repos/thesis/train_scripts/dgx/cifar10/train_shared_4x.sh --use-dali