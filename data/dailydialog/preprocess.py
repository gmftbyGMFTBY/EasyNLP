import random
from itertools import chain
with open('train.txt') as f:
    dataset = [[utterance.strip() for utterance in line.strip().split('__eou__') if utterance.strip()] for line in f.readlines() if line.strip()]
    utterance_datastore = list(chain(*dataset))
    utterance_datastore = list(set(utterance_datastore))
print(f'[!] find {len(utterance_datastore)} utterances')
print(f'[!] find {len(dataset)} samples')
    

with open('train_preprocess.txt', 'w') as f:
    for utterances in dataset:
        negative = random.choice(utterance_datastore)
        positive = '\t'.join(utterances)
        f.write(f'1\t{positive}\n')
        context = '\t'.join(utterances[:-1])
        f.write(f'0\t{context}\t{negative}\n')


