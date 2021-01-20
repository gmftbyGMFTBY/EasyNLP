#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test/train-post> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh train train_generative gpt2 0,1,2,3

# ========== metadata ========== #
max_len=256
seed=50
warmup_ratio=0.1
epoch=5
bsz=16
post_bsz=48
post_epoch=2
post_max_len=512
models=(bert-ft bert-gen bert-gen-ft bert-post dual-bert dual-bert-poly)
ONE_BATCH_SIZE_MODEL=(dual-bert dual-bert-poly)
# ========== metadata ========== #

mode=$1
dataset=$2
model=$3
cuda=$4 

chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    pretrained_model=bert-base-chinese
else
    pretrained_model=bert-base-uncased
fi

if [ $mode = 'init' ]; then
    datasets=(ecommerce douban ubuntu)
    mkdir bak ckpt rest
    for m in ${models[@]}
    do
        for d in ${datasets[@]}
        do
            mkdir -p ckpt/$d/$m
            mkdir -p rest/$d/$m
            mkdir -p bak/$d/$m
        done
    done
elif [ $mode = 'backup' ]; then
    cp ckpt/$dataset/$model/param.txt bak/$dataset/$model/
    cp ckpt/$dataset/$model/best.pt bak/$dataset/$model/
    cp rest/$dataset/$model/rest.txt bak/$dataset/$model/
    cp rest/$dataset/$model/events* bak/$dataset/$model/
elif [ $mode = 'train' ]; then
    ./run.sh backup $dataset $model
    rm ckpt/$dataset/$model/*
    rm rest/$dataset/$model/events*    # clear the tensorboard cache
    
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size $bsz \
        --epoch $epoch \
        --seed $seed \
        --max_len $max_len \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model \
        --warmup_ratio $warmup_ratio
elif [ $mode = 'train-post' ]; then
    # load model parameters from post train checkpoint and fine tuning
    ./run.sh backup $dataset $model
    rm ckpt/$dataset/$model/*
    rm rest/$dataset/$model/events*    # clear the tensorboard cache
    
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size $post_bsz \
        --epoch $post_epoch \
        --seed $seed \
        --max_len $post_max_len \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model \
        --warmup_ratio $warmup_ratio \
        --pretrained_model_path rest/$dataset/bert-post/best_nspmlm.pt
else
    # test
    if [[ ${ONE_BATCH_SIZE_MODEL[@]} =~ $model ]]; then
        bsz=1
    fi
    CUDA_VISIBLE_DEVICES=$cuda python main.py \
        --dataset $dataset \
        --model $model \
        --mode $mode \
        --batch_size $bsz \
        --max_len $max_len \
        --seed $seed \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model
fi
