import csv
import random
from tqdm import tqdm
import json
import ipdb
import sys
import pickle
from collections import Counter
from gensim.summarization import bm25
from elasticsearch import Elasticsearch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', default='lccc', type=str)
    parser.add_argument('--train_size', default=500000, type=int)
    parser.add_argument('--database_size', default=1000000, type=int)
    parser.add_argument('--test_size', default=10000, type=int)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--mode', default='init', type=str)
    parser.add_argument('--samples', default=10, type=int)
    return parser.parse_args()


class ESUtils:

    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch()
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'response': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_max_word'
                    }
                }
            }
            if self.es.indices.exists(index=self.index):
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert_pairs(self, pairs):
        count = self.es.count(index=self.index)['count']
        print(f'[!] begin of the idx: {count}')
        for qa in tqdm(pairs):
            data = {'response': qa}
            self.es.index(index=self.index, body=data)
        print(f'[!] whole database size: {self.es.count(index=self.index)["count"]}')


class ESChat:

    def __init__(self, index_name):
        self.es = Elasticsearch()
        self.index = index_name

    def search(self, query, samples=10):
        dsl = {
            'query': {
                'match': {
                    'response': query
                }
            }
        }
        hits = self.es.search(index=self.index, body=dsl, size=samples)['hits']['hits']
        rest = []
        for h in hits:
            rest.append({
                'score': h['_score'], 
                'response': h['_source']['response']
            })
        return rest


def write_file(dialogs, mode='train', samples=10):
    if mode == 'train':
        with open('train.txt', 'w') as f:
            for context, response in tqdm(dialogs):
                f.write(f'1\t{context}\t{response}\n')
    elif mode == 'test':
        chatbot = ESChat(args['name'])
        with open(f'test.txt', 'w') as f:
            error_counter = 0
            responses = [i[1] for i in dialogs]
            for context, response in tqdm(dialogs):
                rest = [i['response'] for i in chatbot.search(context, samples=samples+1)]
                if response in rest:
                    rest.remove(response)
                if len(rest) >= samples:
                    rest = rest[:samples]
                else:
                    rest.extend(random.sample(responses, samples-len(rest)))
                f.write(f'1\t{context}\t{response}\n')
                for i in rest:
                    f.write(f'0\t{context}\t{i}\n')


def read_file(path, mode='train'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        if mode == 'train':
            data = data['train']
        dialogs = []
        for utterances in data:
            utterances = [''.join(i.split()) for i in utterances]
            context = '\t'.join(utterances[:-1])
            response = utterances[-1]
            dialogs.append((context, response))
    print(f'[!] load {len(dialogs)} samples')
    return dialogs


if __name__ == "__main__":
    args = vars(parse_args())
    random.seed(args['seed'])
    if args['mode'] == 'init':
        train_data = read_file('LCCC-base.json', mode='train')
        train_data_ = random.sample(train_data, args['train_size'])
        write_file(train_data_, mode='train')

        esutils = ESUtils(args['name'], create_index=True)
        train_data_ = random.sample(train_data, args['database_size'])
        responses = [i[1] for i in train_data_]
        esutils.insert_pair(responses)
    elif args['mode'] == 'retrieval':
        test_data = read_file('LCCC-base_test.json', mode='test')
        test_data = random.sample(test_data, args['test_size'])
        write_file(test_data, mode='test', samples=args['samples'])
    else:
        raise Exception(f'Unknow mode: {args["mode"]}')
