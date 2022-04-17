#!/bin/bash
# only for jizhi platform

# read the number of the gpu cards from the config
gpu_num=$(cat jizhi_config.json | python -c "import sys, json; print(json.load(sys.stdin)['host_gpu_num'])")
echo "[!] detected $gpu_num GPUs"

# generate the gpus name sequence
split=','
str=''
for i in $(seq 0 $(($gpu_num-1)));
do
    str=$str$i$split
done
str=${str%?}

# obtain the dataset and model name
names=$(cat jizhi_config.json | python -c "import sys, json; print(json.load(sys.stdin)['readable_name'])")
OLD_IFS=$IFS
IFS='|'
names=($names)
IFS=$OLD_IFS

dataset_name=${names[0]}
model_name=${names[1]}

# train
# echo "RUN $model_name on $dataset_name"
# ./scripts/train.sh $dataset_name $model_name $str

# test on one gpu
# ./scripts/test_rerank.sh $dataset_name $model_name 0
# test the generation model
# ./scripts/test_generation.sh $dataset_name $model_name 0

# inference clean large-scale dataset
./scripts/inference_clean.sh $dataset_name $model_name $str
