#!/bin/bash

dataset=$1
cuda=$2
CUDA_VISIBLE_DEVICES=$cuda python test_copygeneration_deploy.py \
    --dataset $dataset \
    --model copygeneration \
    --recall_topk 20 \
    --partial 1.0 \
    --port 23325
