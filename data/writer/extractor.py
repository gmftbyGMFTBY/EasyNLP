import random
from tqdm import tqdm
import os

# set the seed
random.seed(50)

# load
with open('train_full.txt') as f, open('train.txt', 'w') as fw:
    dataset = []
    for line in tqdm(f.readlines()):
        dataset.append(line)
    dataset = random.sample(dataset, 500000)
    for line in dataset:
        fw.write(line)

with open('test_full.txt') as f, open('test.txt', 'w') as fw:
    dataset = []
    for line in tqdm(f.readlines()):
        dataset.append(line)
    dataset = random.sample(dataset, 10000)
    for line in dataset:
        fw.write(line)
