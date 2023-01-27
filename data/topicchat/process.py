import csv
from itertools import chain
import random

with open('test.csv') as f:
    f = csv.reader(f)
    dataset = []
    next(f)
    cache_id, cache = 0, []
    for cid, utterance, _ in f:
        if cache_id != cid and len(cache) > 0:
            dataset.append(cache)
            cache = [utterance]
        else:
            cache.append(utterance)
        cache_id = cid
    if cache:
        dataset.append(cache)
    utterances = list(chain(*dataset))
    utterances = list(set(utterances))
print(f'[!] collect {len(utterances)} utterances and {len(dataset)} session')

with open('train.txt', 'w') as f:
    for session in dataset:
        context = session[:-1]
        positive = session[-1]
        negative = random.choice(utterances)
        context = '\t'.join(context)
        f.write(f'1\t{context}\t{positive}\n')
        f.write(f'0\t{context}\t{negative}\n')

