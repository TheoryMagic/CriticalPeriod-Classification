#!/bin/bash

# Function to run training with specified GPU and removal epoch
run_training() {
    local gpu=$1
    local deficit_epoch=$2
    
    echo "Starting training on GPU $gpu with removal epoch $deficit_epoch"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --run_name resnet18_blur_de${deficit_epoch} \
        --deficit_epoch ${deficit_epoch} \
        
}

# Array to keep track of background processes
declare -a pids

# Get number of GPUs available
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)

# Counter for GPU assignment
gpu=0

# Loop through removal epochs from 0 to 180 with step 20
for deficit_epoch in $(seq 0 20 180); do
    # Run training on current GPU
    run_training $gpu $deficit_epoch &
    
    # Store process ID
    pids[$gpu]=$!
    
    # Move to next GPU
    gpu=$(( (gpu + 1) % $num_gpus ))
    
    # If we've used all GPUs, wait for one to finish before continuing
    if [ $gpu -eq 0 ]; then
        echo "Waiting for current batch to finish..."
        wait ${pids[@]}
        pids=()
    fi
done

# Wait for any remaining processes to finish
echo "Waiting for final processes to complete..."
wait

echo "All training runs completed!"