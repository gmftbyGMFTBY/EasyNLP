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

# inference
echo "IFNERENCE $model_name on $dataset_name"
# self-play work mode
./scripts/inference_self_play.sh $dataset_name $model_name $str
# response work mode
# ./scripts/inference_response.sh $dataset_name $model_name $str
# context work mode
# ./scripts/inference_context.sh $dataset_name $model_name $str
# extend work mode
# ./scripts/inference_ext_dialog.sh $dataset_name $model_name $str
# bert mask data augmentation
# ./scripts/inference_da.sh $dataset_name $model_name $str
# simcse gray augmentation
# ./scripts/inference_gray_simcse.sh $dataset_name $model_name $str
