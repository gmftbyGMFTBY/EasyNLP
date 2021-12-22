#!/bin/bash

mode=$1
dataset=$2
prefix_name=$3
# --url 11.161.48.51 \
python test_api.py \
    --size 100 \
    --port 22351 \
    --url 9.91.66.241 \
    --mode $mode \
    --dataset $dataset \
    --topk 10 \
    --seed 0 \
    --block_size 1 \
    --prefix_name $prefix_name
