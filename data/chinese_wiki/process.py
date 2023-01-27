'''split the document into multiple chunks'''
import ipdb
from tqdm import tqdm
import json

def load(path):
    with open(path) as f:
        data = []
        for line in tqdm(f.readlines()):
            line = json.loads(line)['q']     
            for idx in range(0, len(line), 512):
                data.append(line[idx:idx+512].strip())
    print(f'[！] collect {len(data)} chunks')
    return data

def write(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            line = json.dumps({'q': line})
            f.write(f'{line}\n')

if __name__ == "__main__":
    data = load('wikitext_summary.txt')
    write(data, 'train.txt')
