#!/bin/bash

# Get today's date in YYYY-MM-DD format
now=$(date +"%Y-%m-%d_%H-%M-%S")

# Name of the log file
logfile="training_log_${now}.log"
#logfile="rn50_eta_cosine_imagenet.log"


# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=6

python -u main.py --model_path None \
                    --data_path /home/wei/data2/Dataset/imagenet/ \
                    --source_dataset IMAGENET \
                    --target_dataset IMAGENET-C \
                    --optimize bn \
                    --tta_batchsize 128 \
                    --severity 5 \
                    --criterion cosine \
                    --network resnet18 \
                    --lr 0.005 \
                    >> "$logfile" 2>&1


python -u main.py --model_path None \
                    --data_path /home/wei/data2/Dataset/imagenet/ \
                    --source_dataset IMAGENET \
                    --target_dataset IMAGENET-C \
                    --optimize bn \
                    --tta_batchsize 128 \
                    --severity 5 \
                    --criterion entropy \
                    --network resnet18 \
                    --lr 0.00025 \
                    >> "$logfile" 2>&1

#batchsize=("8")

#for sever in "${severities[@]}"; do
#for bs in "${batchsize[@]}"; do
#  python -u main.py --model_path None \
#                    --data_path /home/wei/data2/Dataset/imagenet/ \
#                    --source_dataset IMAGENET \
#                    --target_dataset IMAGENET-C \
#                    --optimize bn \
#                    --tta_batchsize ${bs} \
#                    --severity 5 \
#                    --criterion hpl \
#                    --network resnet50 \
#                    --lr 0.00025 \
#                    >> "$logfile" 2>&1
#done

#criterions=("entropy")
#for crit in "${criterions[@]}"; do
#  python -u main.py --model_path None \
#                    --data_path /home/wei/data2/Dataset/imagenet/ \
#                    --source_dataset IMAGENET \
#                    --target_dataset IMAGENET-C \
#                    --optimize bn \
#                    --tta_batchsize 64 \
#                    --severity 5 \
#                    --criterion ${crit} \
#                    --network resnet50 \
#                    --lr 0.00025 \
#                    >> "$logfile" 2>&1
#done

#criterions=("cosine")
#for crit in "${criterions[@]}"; do
#  python -u main.py --model_path None \
#                    --data_path /home/wei/data2/Dataset/imagenet/ \
#                    --source_dataset IMAGENET \
#                    --target_dataset IMAGENET-C \
#                    --optimize bn \
#                    --tta_batchsize 64 \
#                    --severity 5 \
#                    --criterion ${crit} \
#                    --network resnet50 \
#                    --lr 0.005 \
#                    >> "$logfile" 2>&1
#done