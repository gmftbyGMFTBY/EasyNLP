import json
import numpy as np
import ipdb

with open('USR-TC.json') as f:
    data = json.load(f)

dataset = []
for item in data:
    context = '|||'.join(item['context'].split('\n'))
    for response in item['responses']:
        if response['model'] == 'Original Ground Truth':
            continue
        dataset.append({'context': context, 'response': response['response'], 'score': np.mean(response['Overall'])})
print(f'[!] collect {len(dataset)} samples')

with open('test.txt', 'w') as f:
    for item in dataset:
        string = json.dumps(item)
        f.write(f'{string}\n')
