#!/bin/bash

# Naive
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_1x.sh fasterrcnn_resnet50_fpn
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_3x.sh fasterrcnn_resnet50_fpn

# Shared
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_1x.sh fasterrcnn_resnet50_fpn
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_3x.sh fasterrcnn_resnet50_fpn

# Naive
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_1x.sh retinanet_resnet50_fpn
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_single_3x.sh retinanet_resnet50_fpn

# Shared
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_1x.sh retinanet_resnet50_fpn
/home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/run_mps.sh /home/ubuntu/repos/thesis/train_scripts/g5_8xlarge/coco/train_shared_3x.sh retinanet_resnet50_fpn