#!/bin/bash
# only for jizhi platform

# the cuda nums should fit with the jizhi_config.json
source ~/.bashrc
./train.sh douban dual-bert 0,1 
