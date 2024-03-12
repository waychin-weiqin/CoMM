#!/bin/bash

# Get today's date in YYYY-MM-DD format
now=$(date +"%Y-%m-%d_%H-%M-%S")

# Name of the log file
logfile="training_log_${now}_cifar10.log"

# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=1

# Cosine LR=0.02
# Entropy LR = 0.00025

#lr=(0.03 0.04 0.05)
#
#for l in "${lr[@]}"; do
#  python -u main.py --model_path models/Hendrycks2020AugMixWRN_c10.pt \
#                    --data_path /home/wei/data2/Dataset/cifar/ \
#                    --source_dataset cifar-10 \
#                    --target_dataset cifar-10-c \
#                    --optimize bn \
#                    --lr ${l} \
#                    --tta_batchsize 128 \
#                    --severity 5 \
#                    --criterion cosine \
#                    --network wrn-40x2 \
#                    >> "$logfile" 2>&1
#done
python -u main.py --model_path models/Hendrycks2020AugMixWRN_c10.pt \
                  --data_path /home/wei/data2/Dataset/cifar/ \
                  --source_dataset cifar-10 \
                  --target_dataset cifar-10-c \
                  --optimize bn \
                  --lr 0.00025 \
                  --tta_batchsize 128 \
                  --severity 5 \
                  --criterion entropy \
                  --network wrn-40x2 \
#                  >> "$logfile" 2>&1