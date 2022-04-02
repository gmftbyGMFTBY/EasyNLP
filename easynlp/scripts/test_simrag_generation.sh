#!/bin/bash

dataset=$1
model=$2
cuda=$3

CUDA_VISIBLE_DEVICES=$cuda python test_simrag_generation.py \
    --dataset $dataset \
    --model $model \
    --recall_topk 10
