import csv
import random
import ipdb

with open('personality.csv') as f:
    f = csv.reader(f)
    next(f)
    utterance_collection = []
    data = []
    for _, persona, context in f:
        utterances = context.split('\n')
        data.append((persona, utterances))
        utterance_collection.extend(utterances)
    utterance_collection = list(set(utterance_collection))
print(f'[!] collect {len(data)} samples and {len(utterance_collection)} collection')

with open('train.txt', 'w') as f:
    for persona, utterances in data:
        negative = random.choice(utterance_collection)
        context, ground_truth = utterances[:-1], utterances[-1]
        context = '\t'.join(context)
        f.write(f'1\t{persona}\t{context}\t{ground_truth}\n')
        f.write(f'0\t{persona}\t{context}\t{negative}\n')




