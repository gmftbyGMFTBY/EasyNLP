import numpy as np
import ipdb
from itertools import chain

def load(path, train=False):
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if train:
        dataset = []
        for i in range(0, len(lines), 2):
            dataset.append(lines[i:i+2])
    else:
        dataset = []
        for i in range(0, len(lines), 10):
            dataset.append(lines[i:i+10])
    return dataset

def length_count(path):
    with open(path) as f:
        lines = [line.strip().split('\t')[1:] for line in f.readlines() if line.strip()]
        dataset = []
        for i in range(0, len(lines), 10):
            dataset.append(lines[i])

    c_length, r_length = [], []
    for line in dataset:
        c = ''.join(line[:-1])
        c_length.append(len(c))
        r_length.append(len(line[-2]))
    avg_c_l = np.mean(c_length)
    avg_r_l = np.mean(r_length)
    max_c_l = max(c_length)
    min_c_l = min(c_length)
    max_r_l = max(r_length)
    min_r_l = min(r_length)
    print(f'[!] number: {len(dataset)}')
    print(f'[!] avg_c_l: {round(avg_c_l, 4)}')
    print(f'[!] avg_r_l: {round(avg_r_l, 4)}')
    print(f'[!] max_c_l: {round(max_c_l, 4)}')
    print(f'[!] max_r_l: {round(max_r_l, 4)}')
    print(f'[!] min_c_l: {round(min_c_l, 4)}')
    print(f'[!] min_r_l: {round(min_r_l, 4)}')
    return dataset

if __name__ == "__main__":
    length_count('test_10.txt')
