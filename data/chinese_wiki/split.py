'''split the document into multiple chunks'''
import ipdb
import random
from tqdm import tqdm
import json

def load(path):
    with open(path) as f:
        data = []
        for line in tqdm(f.readlines()):
            line = json.loads(line)['q']     
            data.append(line)
    print(f'[！] collect {len(data)} chunks')
    return data

def split(data, training_size):
    length = len(data)
    sample_index = random.sample(range(length), training_size)
    sample_index = set(sample_index)
    sample_data = [data[i] for i in sample_index]
    database_data = [data[i] for i in range(length) if i not in sample_index]
    print(f'[!] sample size: {len(sample_data)}; base size: {len(database_data)}')
    return sample_data, database_data

def write(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            line = json.dumps({'q': line})
            f.write(f'{line}\n')

if __name__ == "__main__":
    data = load('train.txt')
    sample_data, base_data = split(data, 500000)
    write(sample_data, 'sample_data.txt')
    write(base_data, 'base_data.txt')
