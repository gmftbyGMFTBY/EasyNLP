#!/bin/bash

# dude, what the fuck !
export NCCL_IB_DISABLE=1

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
version=$(cat config/base.yaml | shyaml get-value version)

# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_$version.txt

echo "find root_dir: $root_dir"
echo "find version: $version"
echo "write running log into recoder file: $recoder_file"

CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port 28457 train_long.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu 0,1,2,3,4,5,6,7 \
    --total_workers 8 
