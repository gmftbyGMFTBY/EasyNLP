from tqdm import tqdm
from itertools import chain
import random
import ipdb

random.seed(0)

def load_file(path):
    with open(path) as f:
        pos, neg, mid = 0, 0, 0
        dataset = []
        for line_ in tqdm(f.readlines()):
            line = line_.strip().split('\t')
            try:
                score = int(line[-1])
            except:
                continue
            if score == 3:
                pos += 1
            elif score == 1:
                neg += 1
            else:
                mid += 1
            if score == 1 or score == 3:
                dataset.append((1 if score == 3 else 0, line[:-1]))
    print(f'[!] pos|mid|neg: {pos}|{mid}|{neg}')
    print(f'[!] collect {len(dataset)} samples')
    return dataset

def write_file(path, data, mode='train'):
    with open(path, 'w') as f:
        counter = 0
        if mode == 'train':
            for label, utterances in data:
                s = '\t'.join(utterances)
                string = f'{label}\t{s}\n'
                f.write(string)
                counter += 1
        else:
            for label, utterances in data:
                assert label == 1
                negs = random.sample(responses, 9)
                context = utterances[:-1]
                context = '\t'.join(context)
                utterances = [utterances[-1]] + negs
                for idx, u in enumerate(utterances):
                    label = 1 if idx == 0 else 0
                    u = f'{context}\t{u}'
                    f.write(f'{label}\t{u}\n')
                counter += 1
    print(f'[!] write {counter} lines into {path}')


dataset = load_file('qapairs_annotated_all.txt')
dataset = random.sample(dataset, 10000)
true_num = len([item[0] for item in dataset if item[0] == 1])
print(f'[!] ratio: {round(true_num/len(dataset), 4)}')
random.shuffle(dataset)
write_file('test_acc.txt', dataset)
