import csv
import json
import ipdb

with open('ft_local/test.csv' ) as f:
    f = csv.reader(f)
    head = next(f)
    dataset = []
    cache = set()
    for line in f:
        if line[0] not in cache:
            dataset.append({'context': line[0], 'response': line[1], 'score': float(line[2])})
            cache.add(line[0])
    print(f'[!] collect {len(dataset)} samples')

with open('test.txt', 'w') as f:
    for item in dataset:
        string = json.dumps(item)
        f.write(f'{string}\n')
