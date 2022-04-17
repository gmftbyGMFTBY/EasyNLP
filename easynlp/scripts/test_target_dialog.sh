#!/bin/bash

dataset=$1
cuda=$2

CUDA_VISIBLE_DEVICES=$cuda python test_target_dialog.py --dataset $dataset
