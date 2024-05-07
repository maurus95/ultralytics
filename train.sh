#!/bin/bash

# start training
# MP_NUM_THREADS=1
NCCL_P2P_LEVEL=NVL PYTHONPATH=/home/maufri/repos/ultralytics /usr/bin/env /home/maufri/miniconda3/envs/pt-gpu/bin/python \
train.py -d /home/maufri/repos/ultralytics/yolov8_eb_data.yaml --batch_size 32 --device 3 --amp
# train.py -m /home/maufri/repos/ultralytics/runs/detect/train4/weights/last.pt --resume
