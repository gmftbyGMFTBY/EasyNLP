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

./train.sh douban dual-bert $str
