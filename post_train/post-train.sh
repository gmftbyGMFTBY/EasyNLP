#!/bin/bash

# ./post-train.sh <dataset_name> <gpu_ids>
dataset=$1
cuda=$2
chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    ckpt=bert-base-chinese
else
    ckpt=bert-base-uncased
fi

cp -r ../ckpt/$dataset/bert-post/* ../bak/$dataset/bert-post/
cp -r runs ../bak/$dataset/bert-post

echo "[!] back up the checkpoint and logs"

rm -rf ../ckpt/$dataset/bert-post/*
rm -rf runs

echo "[!] clear the checkpoint and logs"

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 post_training.py
