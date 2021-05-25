#!/bin/bash

# check the es index
# curl -X GET localhost:9200/_cat/indices?v

python init_bm25.py \
    --dataset $1 \
    --topk 10 \
    --inner_bsz 128
