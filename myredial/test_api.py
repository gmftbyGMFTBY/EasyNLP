import http.client
import torch
import random
import numpy as np
from tqdm import tqdm
import pprint
import json
import ipdb
from dataloader import *
from config import *
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--block_size', type=int, default=10)
    return parser.parse_args()

def load_fake_rerank_data(path, size=1000):
    # test dataset test
    dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in dataset:
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'context': ' [SEP] '.join(j[0]), 
                        'candidates': [j[1]] + j[2]
                    } for j in cache
                ],
                'lang': 'zh',
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    data = random.sample(data, size)
    return data

def load_fake_recall_data(path, size=1000):
    dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in dataset:
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'str': ' [SEP] '.join(j[0]), 
                        'status': 'editing'
                    } for j in cache
                ],
                'lang': 'zh',
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    data = random.sample(data, size)
    return data

def SendPOST(url, port, method, params):
    headers = {"Content-type": "application/json"}
    conn = http.client.HTTPConnection(url, port)
    conn.request('POST', method, params, headers)
    response = conn.getresponse()
    code = response.status
    reason=response.reason
    data = json.loads(response.read().decode('utf-8'))
    conn.close()
    return data

if __name__ == '__main__':
    args = vars(parser_args())
    args['root_dir'] = '/apdcephfs/share_916081/johntianlan/MyReDial'

    recall_data = load_fake_recall_data(
        f'{args["root_dir"]}/data/writer/test.txt',
        size=args['size'],
    )
    rerank_data = load_fake_rerank_data(
        f'{args["root_dir"]}/data/writer/test.txt',
        size=args['size'],
    )

    # recall test begin
    avg_times = []
    recall_collections = []
    for data in tqdm(recall_data):
        data = json.dumps(data)
        rest = SendPOST('9.91.66.241', 22335, '/recall', data)
        recall_collections.append(rest)
        avg_times.append(rest['header']['core_time_cost_ms'])
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg recall time cost: {avg_t} ms')
    
    # rerank test begin
    avg_times = []
    rerank_collections = []
    for data in tqdm(rerank_data):
        data = json.dumps(data)
        rest = SendPOST('9.91.66.241', 22335, '/rerank', data)
        rerank_collections.append(rest)
        avg_times.append(rest['header']['core_time_cost_ms'])
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms')

    torch.save((recall_collections, rerank_collections), f'{args["root_dir"]}/data/writer/test_api_log.pt')
