from .es_utils import *
from tqdm import tqdm
from config import *
from dataloader.utils import *
import argparse
import json
import ipdb


'''Generate the BM25 gray candidates:
Make sure the q-q BM25 index has been built
'''


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--pool_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--recall_mode', default='q-q', type=str)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--full_turn_length', default=5, type=int)
    return parser.parse_args()

def main_search(args):
    searcher = ESSearcher(
        f'{args["dataset"]}_{args["recall_mode"]}', 
        q_q=True if args['recall_mode']=='q-q' else False
    )

    # load train dataset
    read_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt'

    dataset = read_text_data_utterances_full(read_path, lang=args['lang'], turn_length=full_turn_length)
    data = [(utterances[:-1], utterances[-1]) for label, utterances in dataset if label == 1]
    responses = [utterances[-1] for label, utterances in dataset]
    collector = []
    pbar = tqdm(range(0, len(data), args['batch_size']))
    for idx in pbar:
        # random choice the conversation context to search the topic related responses
        context = [i[0] for i in data[idx:idx+args['batch_size']]]
        response = [i[1] for i in data[idx:idx+args['batch_size']]]
        context_str = [' '.join(i[0]) for i in data[idx:idx+args['batch_size']]]
        rest_ = searcher.msearch(context_str, topk=args['pool_size'])

        rest = []
        for gt_ctx, gt_res, i in zip(context, response, rest_):
            i = list(set(i))
            if gt_res in i:
                i.remove(gt_res)
            if len(i) < args['topk']:
                rest.append(i + random.sample(responses, args['topk']-len(i)))
            else:
                rest.append(i[:args['topk']])

        for q, r, nr in zip(context, response, rest):
            collector.append({'q': q, 'r': r, 'nr': nr})

    with open(write_path, 'w', encoding='utf-8') as f:
        for data in collector:
            string = json.dumps(data)
            f.write(f'{string}\n')


def main_single_search(args):
    q_q_searcher = ESSearcher(
        f'{args["dataset"]}_q-q', 
        q_q=True
    )
    single_searcher = ESSearcher(
        f'{args["dataset"]}_single', 
        q_q=False
    )

    # load train dataset
    read_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt'
    dataset = read_text_data_utterances_full(read_path, lang=args['lang'], turn_length=full_turn_length)
    data = [(utterances[:-1], utterances[-1]) for label, utterances in dataset if label == 1]
    responses = [utterances[-1] for label, utterances in dataset]
    collector = []
    pbar = tqdm(range(0, len(data), args['batch_size']))
    for idx in pbar:
        # random choice the conversation context to search the topic related responses
        context = [i[0] for i in data[idx:idx+args['batch_size']]]
        response = [i[1] for i in data[idx:idx+args['batch_size']]]
        context_str = [' '.join(i[0]) for i in data[idx:idx+args['batch_size']]]
        rest_q_q, rest_single = [], []

        # q-q search
        rest_ = q_q_searcher.msearch(context_str, topk=args['pool_size'])
        for gt_ctx, gt_res, i in zip(context, response, rest_):
            i = list(set(i))
            if gt_res in i:
                i.remove(gt_res)
            if len(i) < args['topk']:
                rest_q_q.append(i + random.sample(responses, args['topk']-len(i)))
            else:
                rest_q_q.append(i[:args['topk']])

        # single search
        query = []
        for item in data[idx:idx+args['batch_size']]:
            ctx, _ = item
            if len(ctx) == 1:
                query.append(ctx[0])
            else:
                query.append(random.choice(ctx[:-1]))
        rest_ = single_searcher.msearch(query, topk=args['pool_size'])
        for gt_ctx, gt_res, i in zip(context, response, rest_):
            i = list(set(i))
            if gt_res in i:
                i.remove(gt_res)
            if len(i) < args['topk']:
                rest_single.append(i + random.sample(responses, args['topk']-len(i)))
            else:
                rest_single.append(i[:args['topk']])

        for q, r, nr, nr_ in zip(context, response, rest_q_q, rest_single):
            collector.append({'q': q, 'r': r, 'q_q_nr': nr, 'single_nr': nr_})

    with open(write_path, 'w', encoding='utf-8') as f:
        for data in collector:
            string = json.dumps(data)
            f.write(f'{string}\n')


if __name__ == '__main__':
    args = vars(parser_args())
    full_turn_length = args['full_turn_length']
    bsz = args['batch_size']
    args['mode'] = 'test'
    args['model'] = 'dual-bert'    # useless
    config = load_config(args)
    args.update(config)
    args['batch_size'] = bsz

    if args['recall_mode'] == 'single':
        main_single_search(args)
    else:
        main_search(args)
