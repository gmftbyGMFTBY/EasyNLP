#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)

# backup
mv $root_dir/ckpt/$dataset/$model/*.pt $root_dir/bak/$dataset/$model
mv $root_dir/rest/$dataset/$model/* $root_dir/bak/$dataset/$model/

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29439 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda
