import csv
from itertools import chain
import random
from tqdm import tqdm
import ipdb

with open('train.csv') as f:
    header = next(f)
    dataset, cache = [], []
    cache_conv_id = None
    counter = 0
    for item in tqdm(f):
        item = item.split(',')
        try:
            assert len(item) == 8
        except:
            ipdb.set_trace()
        conv_id, uid, _, prompt, sid, utterance, _, _ = item

        utterance = utterance.replace('_comma_', ',').strip()
        if cache_conv_id and conv_id != cache_conv_id:
            if cache:
                dataset.append(cache)
            cache = [utterance]
        else:
            cache.append(utterance)
        cache_conv_id = conv_id
    if cache:
        dataset.append(cache)
    print(f'[!] collect {len(dataset)} sessions')

utterances = list(chain(*dataset))
utterances = list(set(utterances))
print(f'[!] collect {len(utterances)} utterances')

# make the response selection train set
with open('train.txt', 'w') as f:
    for session in dataset:
        negative = random.choice(utterances)
        string = '\t'.join(session)
        f.write(f'1\t{string}\n')
        string_ = '\t'.join(session[:-1])
        f.write(f'0\t{string_}\t{negative}\n')
        
