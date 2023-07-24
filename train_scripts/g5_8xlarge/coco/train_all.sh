#!/bin/bash

cpus=$1

# Naive
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_1x.sh fasterrcnn_resnet50_fpn $cpus
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_4x.sh fasterrcnn_resnet50_fpn $cpus

# Shared
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_1x.sh fasterrcnn_resnet50_fpn $cpus
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_4x.sh fasterrcnn_resnet50_fpn $cpus

# Naive
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_1x.sh retinanet_resnet50_fpn $cpus
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_4x.sh retinanet_resnet50_fpn $cpus

# Shared
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_1x.sh retinanet_resnet50_fpn $cpus
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_4x.sh retinanet_resnet50_fpn $cpus