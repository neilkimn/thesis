#!/bin/bash

# Naive
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_1x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_2x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_3x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_4x.sh
##/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_5x.sh
#
## Naive MPS
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_1x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_2x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_3x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_4x.sh
##/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_5x.sh
#
## Shared
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_1x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_2x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_3x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_4x.sh
##/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_5x.sh
#
## Shared MPS
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_1x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_2x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_3x.sh
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_4x.sh
##/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_5x.sh
#
## Naive DALI
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_1x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_2x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_3x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_4x.sh --use-dali
##/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_single_5x.sh --use-dali
#
## Shared DALI
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_1x.sh --use-dali
/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_2x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_3x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_4x.sh --use-dali
#/home/ubuntu/repos/thesis/train_scripts/g5_4xlarge/imagenet_10pct/train_shared_5x.sh --use-dali