#!/bin/bash

mode=$1
dataset=$2
prefix_name=$3
# --url 11.176.92.85 \
# --url 11.145.168.72 \
python test_api.py \
    --size 100 \
    --port 23331 \
    --mode $mode \
    --url 9.91.66.241 \
    --dataset $dataset \
    --topk 10 \
    --seed 0 \
    --block_size 1 \
    --prefix_name $prefix_name
