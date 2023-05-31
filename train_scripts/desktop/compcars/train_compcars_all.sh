#!/bin/bash

# Naive
/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_1x.sh
/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_2x.sh

# Naive MPS
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/kafka/repos/thesis/train_single_scripts/compcars/train_single_1x.sh
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/kafka/repos/thesis/train_single_scripts/compcars/train_single_2x.sh

# Shared
/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh
/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh

# Shared MPS
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh

# Naive DALI
/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_1x.sh --use-dali
/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_2x.sh --use-dali

# Naive MPS DALI
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh "/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_1x.sh --use-dali"
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh "/home/kafka/repos/thesis/train_single_scripts/compcars/train_single_2x.sh --use-dali"

# Shared DALI
/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh --use-dali
/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh --use-dali

# Shared MPS DALI
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh "/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh --use-dali"
/home/kafka/repos/thesis/train_single_scripts/compcars/run_mps.sh "/home/kafka/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh --use-dali"
