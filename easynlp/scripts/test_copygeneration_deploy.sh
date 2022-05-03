#!/bin/bash

cuda=$1
CUDA_VISIBLE_DEVICES=$cuda python test_copygeneration_deploy.py \
    --dataset copygeneration \
    --model copygeneration \
    --recall_topk 20 \
    --port 23331
