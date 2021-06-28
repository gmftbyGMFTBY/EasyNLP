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
    return parser.parse_args()

if __name__ == '__main__':
    args = vars(parser_args())
    bsz = args['batch_size']
    args['mode'] = 'test'
    args['model'] = 'dual-bert'    # useless
    config = load_config(args)
    args.update(config)
    args['batch_size'] = bsz

    searcher = ESSearcher(
        f'{args["dataset"]}_{args["recall_mode"]}', 
        q_q=True if args['recall_mode']=='q-q' else False
    )

    # load train dataset
    read_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'

    dataset = read_text_data_dual_bert(read_path, lang=args['lang'])
    dataset = [(context, response) for label, context, response in dataset if label == 1]
    collector = []
    pbar = tqdm(range(0, len(dataset), args['batch_size']))
    for idx in pbar:
        context = [i[0] for i in dataset[idx:idx+args['batch_size']]]
        response = [i[1] for i in dataset[idx:idx+args['batch_size']]]
        rest = searcher.msearch(context, topk=args['pool_size'])
        rest = [random.sample(i, args['topk']) for i in rest]
        for q, r, nr in zip(context, response, rest):
            collector.append({'q': q, 'r': r, 'nr': nr})

    with open(write_path, 'w') as f:
        for data in collector:
            string = json.dumps(data)
            f.write(f'{string}\n')
