#!/bin/bash

# Naive
/home/neni/repos/thesis/train_single_scripts/compcars/train_single_1x.sh
/home/neni/repos/thesis/train_single_scripts/compcars/train_single_2x.sh

# Naive MPS
/home/neni/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/neni/repos/thesis/train_single_scripts/compcars/train_single_1x.sh
/home/neni/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/neni/repos/thesis/train_single_scripts/compcars/train_single_2x.sh

# Shared
/home/neni/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh
/home/neni/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh

# Shared MPS
/home/neni/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/neni/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh
/home/neni/repos/thesis/train_single_scripts/compcars/run_mps.sh /home/neni/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh

# Naive DALI
/home/neni/repos/thesis/train_single_scripts/compcars/train_single_1x.sh --use-dali
/home/neni/repos/thesis/train_single_scripts/compcars/train_single_2x.sh --use-dali

# Shared DALI
/home/neni/repos/thesis/train_single_scripts/compcars/train_shared_1x.sh --use-dali
/home/neni/repos/thesis/train_single_scripts/compcars/train_shared_2x.sh --use-dali