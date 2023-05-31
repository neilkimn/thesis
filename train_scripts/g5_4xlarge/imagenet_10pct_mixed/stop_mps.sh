#!/bin/bash

echo quit | nvidia-cuda-mps-control

sudo nvidia-smi -i 0 -c DEFAULT