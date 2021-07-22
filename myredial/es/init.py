from config import *
from dataloader.utils import *
from .es_utils import *
import argparse
import random
import ipdb

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--recall_mode', default='q-r', type=str, help='q-q/q-r')
    return parser.parse_args()

if __name__ == "__main__":
    args=  vars(parser_args())
    args['mode'] = 'test'
    args['model'] = 'dual-bert'
    config = load_config(args)
    args.update(config)
    print('test', args)

    random.seed(args['seed'])

    train_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    extend_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_unparallel.txt'
    # test_path = f'{args["root_dir"]}/data/{args["dataset"]}/test.txt'

    train_data = load_qa_pair(train_path, lang=args['lang'])
    # test_data = load_qa_pair(test_path, lang=args['lang'])
    extend_data = load_qa_pair_extend(extend_data, lang=args['lang'])
    data = train_data + extend_data

    builder = ESBuilder(
        f'{args["dataset"]}_{args["recall_mode"]}',
        create_index=True,
        q_q=True if args['recall_mode'] == 'q-q' else False,
    )

    builder.insert(data)
