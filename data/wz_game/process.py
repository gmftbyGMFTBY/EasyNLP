from tqdm import tqdm
import random
random.seed(0)

with open('test_.txt') as f:
    dataset = []
    for line in f.readlines():
        key, s1, s2 = line.strip().split('\t')
        dataset.append((key, s1, s2))

rand_num = 9
with open('test.txt', 'w') as f:
    for key, s1, s2 in tqdm(dataset):
        cands = []
        for i in range(rand_num):
            while True:
                k, s3, _ = random.choice(dataset)
                if k != key:
                    break
            cands.append(s3)
        for idx, i in enumerate([s2] + cands):
            label = 1 if idx == 0 else 0
            f.write(f'{s1}\t{i}\t{label}\n')
