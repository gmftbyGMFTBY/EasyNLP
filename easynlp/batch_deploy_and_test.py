import sys
import os
import subprocess
import time

'''deploy the tasks on 8 gpus and activate 8 task to post the requests'''

nums = 8
# deploy the server
for i in range(nums):
    cmd_str = f'./scripts/deploy.sh {i} &'
    subprocess.call(cmd_str, shell=True)
    print(f'[!] run the server: {cmd_str}')

time.sleep(30)
# deploy the asker
for i in range(nums):
    port = 22330 + i
    cmd_str = f'python test_api.py --url 9.91.66.241 --port {port} --mode pipeline --dataset restoration-200k --topk 100 --seed 0 --block_size 1 --worker_num 8 --worker_id {i} --prefix_name colbert'
    subprocess.call(cmd_str, shell=True)
    print('[!] run the poster: \n', cmd_str)

