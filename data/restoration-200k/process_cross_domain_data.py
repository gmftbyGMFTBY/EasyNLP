import random
import ipdb

with open('train.txt') as f:
    dataset = [line.strip().split('\t') for line in f.readlines()]
    dataset = [dataset[i:i+2] for i in range(0, len(dataset), 2)]

data = []
for positive, negative in dataset:
    assert positive[1] == negative[1]
    assert positive[0] == '1' and negative[0] == '0'
    data.append((positive[1:-1], positive[-1], negative[-1]))

with open('train_cross_domain.txt', 'w') as f:
    counter = 0
    for ctx, p, n in data:
        ctx = '\t'.join(ctx)
        string = f'{counter}\t{ctx}\t{p}\t{n}\n'
        f.write(string)
        counter += 1
