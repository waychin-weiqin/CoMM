#!/bin/bash

# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

python -u main.py --model_path models/Hendrycks2020AugMixWRN_c100.pt \
                  --data_path /home/wei/data2/Dataset/cifar/ \
                  --source_dataset cifar-100 \
                  --target_dataset cifar-100-c \
                  --lr 0.005 \
                  --tta_batchsize 128 \
                  --severity 5 \
                  --criterion entropy \
                  --network wrn-40x2
