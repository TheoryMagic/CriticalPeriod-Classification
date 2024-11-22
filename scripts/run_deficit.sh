#!/bin/bash

# 定义缺陷类型数组
deficits=("blur" "vertical_flip" "label_permutation" "noise")
# 定义移除epoch的数组，例如每隔20个epoch移除一次
deficit_epochs=(0 20 40 60 80 100 120 140 160 180)

# 获取可用GPU数量
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo "Number of GPUs available: $num_gpus"

# 函数：运行训练
run_training() {
    local gpu=$1
    local deficit_type=$2
    local deficit_epoch=$3
    local run_name=${deficit_type}_de${deficit_epoch}

    echo "Starting training: Deficit=$deficit_type, Removal Epoch=$deficit_epoch on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --run_name $run_name \
        --deficit_type $deficit_type \
        --deficit_epoch $deficit_epoch \
        --project CLP_Adam &  # 添加新的项目名称
}

# 数组存储后台进程的PID
declare -a pids

# GPU计数器
gpu=0

# 遍历每种缺陷类型
for deficit in "${deficits[@]}"; do
    echo "Processing deficit type: $deficit"
    # 遍历每个移除epoch
    for epoch in "${deficit_epochs[@]}"; do
        run_training $gpu $deficit $epoch
        # 存储PID
        pids[$gpu]=$!
        # 切换GPU
        gpu=$(( (gpu + 1) % num_gpus ))
        # 如果所有GPU都在使用，等待其中一个完成
        if [ $gpu -eq 0 ]; then
            echo "Waiting for current batch to finish..."
            wait ${pids[@]}
            pids=()
        fi
    done
done

# 等待所有后台进程完成
echo "Waiting for all training runs to complete..."
wait

echo "All training runs completed!"
