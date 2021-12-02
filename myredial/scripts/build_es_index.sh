#!/bin/bash

dataset=$1
recall_mode=$2
python -m es.init \
    --dataset $dataset \
    --recall_mode $recall_mode \
    --maximum_sentence_num 1000000
