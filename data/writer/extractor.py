import random
from tqdm import tqdm
import os

# load
with open('train_full.txt') as f, open('train.txt', 'w') as fw:
    dataset = []
    for line in tqdm(f.readlines()):
        dataset.append(line)
    dataset = random.sample(dataset, 500000)
    for line in dataset:
        fw.write(line)

