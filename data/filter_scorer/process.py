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
responses = list(chain(*[item[1] for item in dataset]))
test_size = 1000
pos_index = [idx for idx, (label, _) in enumerate(dataset) if label == 1]
test_index = set(random.sample(pos_index, test_size))
test_set = [dataset[i] for i in test_index]
train_set = [dataset[i] for i in range(len(dataset)) if i not in test_index]

random.shuffle(train_set)
random.shuffle(test_set)

write_file('train.txt', train_set)
write_file('test.txt', test_set, mode='test')
