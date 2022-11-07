import json
import random
import numpy as np
import ipdb
import torch

'''filter the useful hard targets from the TopKG, using the keyword-memory information of our methods'''

k2m = torch.load('k2m.pt')
hard_targets = json.load(open('hard_target_list.json'))
print(f'[!] load {len(hard_targets)}')

min_size, max_size = 10, 50

valid_keywords = []
for hard in hard_targets:
    if hard in k2m:
        valid_keywords.append((hard, len(k2m[hard])))

k2m_counts = [(key, len(value)) for key, value in k2m.items()]
k2m_filter = sorted(k2m_counts, key=lambda x:x[1])
min_index, max_index = 0, 0
previous_size = -1
for idx, (key, size) in enumerate(k2m_filter):
    if previous_size < min_size and size == min_size:
        min_index = idx
    if previous_size == max_size and size > max_size:
        max_index = idx
    previous_size = size
k2m_filter = k2m_filter[min_index:max_index]

# load the tgcp test set dataset
with open('tgcp_test.txt') as f:
    dataset = []
    for line in f.readlines():
        utterance, keyword = line.strip().split('\t')
        dataset.append(utterance)
    utterances = list(set(dataset))
    print(f'[!] load {len(utterances)} initial sentences as the begining')

# build the benchmark
random.seed(0)
dataset = []
for keyword, _ in k2m_filter:
    utterance = random.choice(utterances)
    dataset.append((utterance, keyword))

# save into the file
with open('tgcp_hard_mine.txt', 'w') as f:
    for utterance, keyword in dataset:
        string = f'{utterance}\t{keyword}\n'
        f.write(string)

# print(f'[!] valid keywords: {len(valid_keywords)}')
# counts = [i for _, i in valid_keywords]
# 
# print(f'[!] min: {min(counts)}; max: {max(counts)}; mean: {np.mean(counts)}')
