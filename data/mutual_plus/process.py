from tqdm import tqdm
import ipdb
import pickle
import os
import json

def process_one_file(path, test=False):
    with open(path) as f:
        data = json.load(f)
    convert_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    article = data['article']
    article = article.replace('m :', '[M]').replace('f :', '[F]')

    options = []
    for op in data['options']:
        op = op.replace('m :', '[M]').replace('f :', '[F]')
        options.append(op)

    data['article'] = article
    data['options'] = options

    if test is False:
        data['answers'] = convert_map[data['answers']]
    return data

def collect_one_mode(mode):
    dataset = []
    for file in tqdm(os.listdir(mode)):
        if file.endswith('.txt') is False:
            continue
        path = os.path.join(mode, file)
        data = process_one_file(path, test=mode=='test')
        dataset.append(data)
    print(f'[!] load {len(dataset)} files')
    return dataset

if __name__ == "__main__":
    # train set
    train_set = collect_one_mode('train')
    test_set  = collect_one_mode('test')
    dev_set   = collect_one_mode('dev')
    print(f'[!] load {len(train_set) + len(test_set) + len(dev_set)} sessions')

    # save 
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    with open('test.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    with open('dev.pkl', 'wb') as f:
        pickle.dump(dev_set, f)
