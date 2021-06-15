import random
import ipdb
from tqdm import tqdm
import os
import json

# set the seed
random.seed(50)

with open('train.txt') as f, open('inference.txt', 'w') as fw:
    responses = []
    for line in tqdm(f.readlines()):
        line = json.loads(line.strip())
        responses.extend(line['q'])
        responses.append(line['r'])
    responses = [i.strip() for i in responses if i.strip()]
    responses = list(set(responses))
    print(f'[!] collect {len(responses)} utterances')

    for response in responses:
        fw.write(f'{response.strip()}\n')

