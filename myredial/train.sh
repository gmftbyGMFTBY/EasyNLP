#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

rm ckpt/$dataset/$model/*
rm rest/$dataset/$model/events*    # clear the tensorboard cache

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
echo "redirect the output: ${root_dir}/log/${dataset}/${model}/log.txt"

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29406 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda > ${root_dir}/log/${dataset}/${model}/log.txt
