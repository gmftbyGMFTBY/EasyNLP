#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test/train-post> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh train train_generative gpt2 0,1,2,3

# ========== metadata ========== #
max_len=256
res_max_len=128
seed=50
warmup_ratio=0.1
epoch=10
bsz=48
head_num=5     # hyperparameter of the dual-bert-one2mnay: 11 heads means there are 1 groundtruths and 10 retrieved candidates
pre_extract=50
inf_bsz=64
# 
post_bsz=48
post_epoch=10
post_max_len=256
post_res_max_len=128
models=(sa-bert bert-ft bert-ft-multi bert-gen bert-gen-ft bert-post dual-bert-gen dual-bert dual-bert-poly dual-bert-cl dual-bert-vae dual-bert-vae2 dual-bert-one2many dual-bert-hierarchical dual-bert-mb dual-bert-adv dual-bert-jsd)
ONE_BATCH_SIZE_MODEL=(bert-ft-multi dual-bert dual-bert-poly dual-bert-cl dual-bert-gen dual-bert-vae dual-bert-vae2 dual-bert-one2many dual-bert-hierarchical dual-bert-mb dual-bert-adv dual-bert-jsd)
datasets=(ecommerce douban ubuntu lccc lccc-large)
chinese_datasets=(douban ecommerce lccc lccc-large)
# ========== metadata ========== #

mode=$1
dataset=$2
model=$3
cuda=$4 

if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    pretrained_model=bert-base-chinese
    lang=zh
    # pretrained_model=hfl/chinese-bert-wwm
else
    pretrained_model=bert-base-uncased
    lang=en
fi

if [ $mode = 'init' ]; then
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
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29401 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size $bsz \
        --epoch $epoch \
        --seed $seed \
        --max_len $max_len \
        --res_max_len $res_max_len \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model \
        --head_num $head_num \
        --lang $lang \
        --warmup_ratio $warmup_ratio
elif [ $mode = 'inference' ]; then
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
        --dataset $dataset \
        --model $model \
        --mode inference \
        --batch_size $inf_bsz \
        --seed $seed \
        --max_len $max_len \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model \
        --lang $lang \
        --warmup_ratio $warmup_ratio
    # reconstruct
    python model/searcher.py \
        --dataset $dataset \
        --nums ${#gpu_ids[@]} \
        --inner_bsz 128 \
        --pre_extract $pre_extract \
        --topk 10
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
        --res_max_len $post_res_max_len \
        --multi_gpu $cuda \
        --pretrained_model $pretrained_model \
        --warmup_ratio $warmup_ratio \
        --pretrained_model_path ckpt/$dataset/bert-post/best_nspmlm.pt
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
        --res_max_len $res_max_len \
        --seed $seed \
        --multi_gpu $cuda \
        --head_num $head_num \
        --lang $lang \
        --pretrained_model $pretrained_model
fi
