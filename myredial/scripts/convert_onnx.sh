#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
version=$(cat config/base.yaml | shyaml get-value version)

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python convert_to_onnx.py \
    --dataset $dataset \
    --model $model
