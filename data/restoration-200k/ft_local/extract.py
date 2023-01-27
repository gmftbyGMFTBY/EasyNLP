import ipdb

dataset = []
with open('test.txt') as f:
    for line in f.readlines():
        items = line.split('\t')
        dataset.append(items[:-3])
print(f'[!] load {len(dataset)} sessions')

with open('new_test.txt' , 'w') as f:
    for line in dataset:
        f.write('\t'.join(line) + '\n')
