#!/bin/bash
# only for jizhi platform

# the cuda nums should fit with the jizhi_config.json
source ~/.bashrc
./run.sh train writer dual-bert-gray 6,7 
