#!/bin/bash

# Naive
/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_1x.sh
/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_2x.sh
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_3x.sh
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_4x.sh

# Shared
/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_1x.sh
/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_2x.sh
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_3x.sh
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_4x.sh

# Naive DALI
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_1x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_2x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_3x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_single_4x.sh --use-dali

# Shared DALI
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_1x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_2x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_3x.sh --use-dali
#/home/kafka/repos/thesis/train_scripts/desktop/imagenet/train_shared_4x.sh --use-dali