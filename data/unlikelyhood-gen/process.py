from tqdm import tqdm
import numpy as np
import re
import ipdb
import random

def load_data(path):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            dataset.append(line)
    return dataset

def write_data(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write(f'{item}\n')

def clean(data):
    dataset = []
    bad_words = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for item in tqdm(data):
        sentences_ = re.split('(。|，|？|；|！|、)', item)
        sentences = []
        for s in sentences_:
            if len(sentences) == 0:
                sentences.append(s)
                continue
            if sentences[-1] in ['。', '，', '？', '；', '！', '、']:
                sentences[-1] += s
            else:
                sentences.append(s)
        if np.mean([len(i) for i in sentences]) < 6:
            continue

        counter = 0
        for char in item:
            if char in bad_words:
                counter += 1
        if len(item) > 500 or counter / len(item) > 0.3:
            continue
        else:
            dataset.append(item)
    print(f'[!] overlap: {len(dataset)}/{len(data)}')
    return dataset


if __name__ == "__main__":
    random.seed(0)
    data = load_data('train_.txt')
    random.shuffle(data)
    train_data = data[:1000000]
    test_data = clean(data[1000000:])
    # test_data = random.sample(test_data, 1200)
    write_data('train.txt', data)
    write_data('test_.txt', test_data)
