import json
import random

random.seed(0)

with open('concepts_nv.json') as f:
    dataset = []
    for line in f.readlines():
        item = json.loads(line)
        target = item['hard_target']
        context = item['dialog'][:1]
        dataset.append((target, context))

with open('hard_target.txt', 'w') as f:
    for target, context in dataset:
        string = '\t'.join(context) + '\t' + target + '\n'
        f.write(string)
