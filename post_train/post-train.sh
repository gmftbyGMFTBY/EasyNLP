#!/bin/bash

# ./post-train.sh <dataset_name> <nspmlm/mlm> <gpu_ids>
dataset=$1
model=$2
cuda=$3

chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    ckpt=bert-base-chinese
else
    ckpt=bert-base-uncased
fi

if [ $model = 'mlm' ]; then
    save_path=ckpt/$dataset/bert-post/best_mlm.pt
elif [ $model = 'nspmlm' ]; then
    save_path=ckpt/$dataset/bert-post/best_nspmlm.pt
elif [ $model = 'nsp' ]; then
    save_path=ckpt/$dataset/bert-post/best_nsp.pt
else
    echo "[!] wrong post training mode: $model"
    exit
fi

cp -r ckpt/$dataset/bert-post/* bak/$dataset/bert-post/
echo "[!] back up the checkpoint and logs"

rm -rf ckpt/$dataset/bert-post/*
echo "[!] clear the checkpoint and logs"

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 post_training.py \
    --dataset $dataset \
    --bert_pretrained $ckpt \
    --data_path data/$dataset/train_post.hdf5 \
    --save_path $save_path \
    --warmup_ratio 0.1 \
    --seed 50 \
    --grad_clip 5 \
    --batch_size 16 \
    --epoch 2 \
    --lr 3e-5 \
    --model $model
