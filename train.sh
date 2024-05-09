#!/bin/bash

# start training
# MP_NUM_THREADS=1
NCCL_P2P_LEVEL=NVL PYTHONPATH=/home/maufri/repos/ultralytics /usr/bin/env /home/maufri/miniconda3/envs/pt-gpu/bin/python \
train.py -m yolov8l.yaml -d /home/maufri/repos/ultralytics/etram_ts.yaml --batch_size 64 --device 0,1 --amp --optimizer SGD
# train.py -m runs/detect/train4/weights/best.pt -d /home/maufri/repos/ultralytics/etram_histo.yaml --batch_size 64 --device 2,3 --amp --optimizer SGD
# train.py -m /home/maufri/repos/ultralytics/runs/detect/train/weights/last.pt --resume
