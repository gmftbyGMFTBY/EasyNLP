#!/bin/bash

mode=$1
dataset=$2
python test_api.py \
    --size 1000 \
    --url 9.91.66.241 \
    --port 22335 \
    --mode $mode \
    --dataset $dataset \
    --block_size 1
