import os
import random
from tqdm import tqdm
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import qparser
from jieba.analyse import ChineseAnalyzer
import json
import argparse

def load_dataset(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            utterance = ''.join(line[-1].split())
            dataset.append(utterance)
    dataset = list(set(dataset))
    print(f'[!] read {len(dataset)} responses from {path}')
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='douban')
    args = vars(parser.parse_args())

    path = f'data/{args["dataset"]}/train.txt'
    dataset = load_dataset(path)

    if args['dataset'] in ['douban', 'ecommerce']:
        schema = Schema(
            response=TEXT(stored=True, analyzer=ChineseAnalyzer()),
        )
    else:
        schema = Schema(
            response=TEXT(stored=True),
        )

    indexdir = f'indexdir/{args["dataset"]}'
    if not os.path.exists(indexdir):
        os.mkdir(indexdir)
    ix = create_in(indexdir, schema)

    writer = ix.writer()
    for response in tqdm(dataset):
        writer.add_document(response=response)
    writer.commit()

    searcher = ix.searcher()
    parser = QueryParser('response', schema=ix.schema, group=qparser.OrGroup)
    q = parser.parse('我特别喜欢你戴眼镜的样子')
    print(q)
    results = searcher.search(q)
    print(f'{len(results)}')
    for i in range(len(results)):
        print(results[i].fields())

