#!/bin/bash

mode=$1
dataset=$2
prefix_name=$3
# --url 9.91.66.241 \
python test_api.py \
    --size 100 \
    --port 22351 \
    --url 11.145.186.5 \
    --mode $mode \
    --dataset $dataset \
    --topk 10 \
    --seed 0 \
    --block_size 1 \
    --prefix_name $prefix_name
