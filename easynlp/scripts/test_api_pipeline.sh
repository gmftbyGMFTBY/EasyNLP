#!/bin/bash

# size is 500, because the test size is 500 in restoration-200k corpus test set
dataset=$1
prefix_name=$2
python test_api.py \
    --url 9.91.66.241 \
    --port 23331 \
    --mode pipeline \
    --dataset $dataset \
    --topk 100 \
    --seed 0 \
    --block_size 1 \
    --worker_num 1 \
    --worker_id 0 \
    --prefix_name $prefix_name
