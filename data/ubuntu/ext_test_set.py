from tqdm import tqdm
from copy import deepcopy
import random
import ipdb

dataset = []
responses = []
with open('test_10.txt') as f:
    cache = []
    idx = 0
    for line in f.readlines():
        line = line.strip().split('\t')
        responses.extend(line[1:])
        if idx == 0:
            cache.append(line)
            idx += 1
            continue
        if idx % 10 == 0:
            dataset.append(cache)
            cache = [line]
        else:
            cache.append(line)
        idx += 1
    if cache:
        dataset.append(cache)
responses = [i.strip() for i in responses if i.strip()]
dataset = random.sample(dataset, 1000)

print(f'[!] collect {len(dataset)} sessions and {len(responses)} utterances')

for num in [50, 100, 1000]:
    nd = []
    with open(f'test_{num}.txt', 'w') as f:
        for session in deepcopy(dataset):
            samples = random.sample(responses, num-10)
            ctx = session[0][1:-1]
            neg_session = [['0'] + ctx + [res] for res in samples]
            session.extend(neg_session)
            try:
                assert len(session) == num
            except:
                ipdb.set_trace()
            for sample in session:
                string = '\t'.join(sample)
                f.write(f'{string}\n')

