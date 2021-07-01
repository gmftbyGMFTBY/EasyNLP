#!/bin/bash

dataset=$1
recall_mode=$2
python -m es.init \
    --dataset $dataset \
    --model dual-bert \
    --recall_mode $recall_mode
