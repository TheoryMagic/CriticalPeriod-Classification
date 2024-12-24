#!/bin/bash

# 定义学习率列表
learning_rates=(0.001 0.01)

# 定义缺陷类型数组
deficits=("blur" "vertical_flip" "label_permutation" "noise")

# 定义移除epoch的数组
deficit_epochs=(0 20 40 60 80 100 120 140 160 180)

num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo "Number of GPUs available: $num_gpus"

run_training() {
    local gpu=$1
    local lr=$2
    local deficit_type=$3
    local deficit_epoch=$4
    local run_name=${deficit_type}_lr${lr}_de${deficit_epoch}

    echo "Starting training: LR=$lr, Deficit=$deficit_type, Removal Epoch=$deficit_epoch on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --lr $lr \
        --run_name $run_name \
        --deficit_type $deficit_type \
        --deficit_epoch $deficit_epoch \
        --project CLP_RMSProp &  # 改为新项目 CLP_RMSProp
}

declare -a pids

gpu=0

for lr in "${learning_rates[@]}"; do
    echo "Processing learning rate: $lr"
    for deficit in "${deficits[@]}"; do
        echo "  Deficit type: $deficit"
        for epoch in "${deficit_epochs[@]}"; do
            run_training $gpu $lr $deficit $epoch
            pids[$gpu]=$!
            gpu=$(( (gpu + 1) % num_gpus ))
            if [ $gpu -eq 0 ]; then
                echo "Waiting for current batch to finish..."
                wait "${pids[@]}"
                pids=()
            fi
        done
    done
done

echo "Waiting for all training runs to complete..."
wait
echo "All training runs completed!"
