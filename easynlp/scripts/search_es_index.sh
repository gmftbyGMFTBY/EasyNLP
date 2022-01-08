#!/bin/bash

# q-q is the default recall method

dataset=$1
recall_mode=$2
python -m es.bm25_gray \
    --dataset $dataset \
    --pool_size 20 \
    --topk 10 \
    --batch_size 128 \
    --full_turn_length 100 \
    --recall_mode $recall_mode
