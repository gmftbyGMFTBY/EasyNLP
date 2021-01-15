#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh train train_generative gpt2 0,1,2,3

mode=$1
dataset=$2
model=$3
cuda=$4 

chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    lang=bert-base-chinese
else
    lang=bert-base-uncased
fi

if [ $mode = 'init' ]; then
    models=(bert-adapt bert-ft bert-gen bert-gen-ft bert-post)
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
        --batch_size 48 \
        --epoch 5 \
        --seed 50 \
        --max_len 256 \
        --multi_gpu $cuda \
        --lang $lang \
        --warmup_ratio 0.1
else
    CUDA_VISIBLE_DEVICES=$cuda python main.py \
        --dataset $dataset \
        --model $model \
        --mode $mode \
        --batch_size 5 \
        --max_len 256 \
        --seed 50 \
        --multi_gpu $cuda \
        --lang $lang
fi
