from tqdm import tqdm

with open('dev.txt') as f:
    dataset = [i.strip() for i in f.readlines()]

with open('test.txt', 'w') as f:
    for line in dataset[:500]:
        f.write(f'{line}\n')
