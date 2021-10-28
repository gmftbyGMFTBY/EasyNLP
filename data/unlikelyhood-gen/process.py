from tqdm import tqdm
import ipdb
import random
import os
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=1000000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--seed', type=float, default=0.0)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=int, default=32)
    return parser.parse_args()


def load(path):
    dataset = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        while len(dataset) < args['train_size'] + args['test_size']:
            line = f.readline()
            dataset.append(line.strip())
            if len(dataset) % 10000 == 0:
                print(f'[!] lines: {len(dataset)}', end='\r')
    dataset = split_the_passage(dataset)
    return dataset


def split_the_passage(dataset):
    nd = []
    for passage in tqdm(dataset):
        passage = ''.join(passage.split())
        for i in range(0, len(passage), args['max_length']):
            sample = passage[i:i+args['max_length']]
            if len(sample) < args['min_length']:
                continue
            nd.append(sample)
    print(f'[!] collect {len(nd)} samples for training')
    return nd

def collect(index, dataset):
    counter = 0
    n_dataset = []
    for idx in index:
        try:
            n_dataset.append(dataset[idx])
        except Exception as error:
            counter += 1
    print(f'[!] get {len(dataset)} documents; find {counter} errors')
    return n_dataset

def write(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(f"{line}\n")
    print(f'[!] write {len(data)} samples into {path}')


if __name__ == "__main__":
    args = vars(parser_args())
    random.seed(args['seed'])
    dataset = load('train.txt07')
    length = len(dataset)
    print(f'[!] find {length} samples in the file')
    train_idx = random.sample(range(length), args['train_size'])
    test_idx = random.sample(list(set(range(length)) - set(train_idx)), args['test_size'])
    # collect
    train_dataset = collect(train_idx, dataset)
    test_dataset = collect(test_idx, dataset)

    write(train_dataset, 'train.txt')
    write(test_dataset, 'test.txt')
    

    
