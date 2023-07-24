#!/bin/bash

#taskset --cpu-list 0-3 systemd-run --scope -p MemoryMax=16G ./train_all.sh 3

taskset --cpu-list 0-7 systemd-run --scope -p MemoryMax=32G ./train_all.sh 7

taskset --cpu-list 0-15 systemd-run --scope -p MemoryMax=64G ./train_all.sh 15

./train_all.sh 31