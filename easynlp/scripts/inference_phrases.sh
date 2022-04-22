#!/bin/bash
export NCCL_IB_DISABLE=1

dataset=$1
model=$2
cuda=$3

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29304 inference.py \
    --dataset $dataset \
    --model $model \
    --nums ${#gpu_ids[@]} \
    --work_mode phrases \
    --cut_size 250000 \
    --train_size 5000000 \
    --overall_size 10000000
