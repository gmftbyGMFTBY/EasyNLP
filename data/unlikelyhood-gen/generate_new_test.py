from tqdm import tqdm
import re
import ipdb
import random
import torch

def load_data(path):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            dataset.append(line)
    return dataset

def process_test_data(data):
    new_dataset = []
    for text in data:
        prefix = text[:prefix_len]
        test_pos = text[prefix_len:]
        # test negative
        sentences_ = re.split('(。|，|？|；|！|、)', test_pos)
        sentences = []
        for s in sentences_:
            if len(sentences) == 0:
                sentences.append(s)
                continue
            if sentences[-1] in ['。', '，', '？', '；', '！', '、']:
                sentences[-1] += s
            else:
                sentences.append(s)
        random.shuffle(sentences)
        test_neg = ''.join(sentences)
        if test_neg and test_pos and prefix:
            new_dataset.append((prefix, test_pos, test_neg))
    return new_dataset

def write_data(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write(f'{item}\n')

if __name__ == "__main__":
    random.seed(0)
    prefix_len = 128
    data = load_data('test_.txt')
    data = process_test_data(data)
    data = random.sample(data, 1000)
    torch.save(data, 'test.pt')
