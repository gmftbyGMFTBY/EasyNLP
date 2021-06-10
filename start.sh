#!/bin/bash
# only for jizhi platform

# the cuda nums should fit with the jizhi_config.json
./train.sh douban dual-bert 0,1,2,3,4,5,6,7 > log.txt 
