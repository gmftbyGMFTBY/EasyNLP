from tqdm import tqdm
import ipdb
import json

def read_file(path, mode='train'):
    with open(path) as f:
        if mode == 'train':
            data = json.load(f)['train']
        else:
            data = json.load(f)
        dialogs = []
        for i in tqdm(data):
            i = [''.join(j.split()) for j in i]
            dialogs.append(i)
    print(f'[!] collect {len(data)} samples for {mode} mode ...')
    return dialogs

def write_file(dataset, path):
    with open(path, 'w') as f:
        for data in tqdm(dataset):
            str_ = '\t'.join(data)
            f.write(f'1\t{str_}\n')

if __name__ == '__main__':
    dataset = read_file('LCCC-base.json', mode='train')
    write_file(dataset, 'train.txt')
    dataset = read_file('LCCC-base_test.json', mode='test')
    write_file(dataset, 'test.txt')
