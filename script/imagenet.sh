#!/bin/bash

# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

python -u main.py --model_path None \
                  --data_path /home/wei/data2/Dataset/imagenet/ \
                  --source_dataset IMAGENET \
                  --target_dataset IMAGENET-C \
                  --tta_batchsize 128 \
                  --severity 5 \
                  --criterion cosine \
                  --network resnet18 \
                  --lr 0.005 \
