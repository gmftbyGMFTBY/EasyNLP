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
    parser.add_argument('--size', type=int, default=500)
    parser.add_argument('--block_size', type=int, default=10)
    parser.add_argument('--topk', type=int, default=10, help='topk candidates for recall')
    parser.add_argument('--mode', type=str, default='rerank/recall/pipeline')
    parser.add_argument('--url', type=str, default='9.91.66.241')
    parser.add_argument('--port', type=int, default=22335)
    parser.add_argument('--dataset', type=str, default='douban')
    parser.add_argument('--seed', type=float, default=0.0)
    parser.add_argument('--prefix_name', type=str, default='')
    # worker from 0-7, only for bert-ft full-rank
    parser.add_argument('--worker_num', type=int, default=1)
    parser.add_argument('--worker_id', type=int, default=0)
    return parser.parse_args()

def load_pipeline_data_with_worker_id(path, size=1000):
    '''for pipeline and recall test'''
    data = read_text_data_utterances(path, lang='zh')
    dataset = []
    for i in range(0, len(data), 10):
        session = data[i:i+10]
        cache = []
        for label, utterances in session:
            if label == 1:
                cache.append(utterances[-1])
        # NOTE:
        dataset.append({
            'ctx': utterances[:-1],
            'res': cache
        })
    # obtain a part of the data
    size = int(len(dataset) / args['worker_num'])
    dataset = dataset[size*args['worker_id']:size*(args['worker_id'] + 1)]

    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    collector = []
    for item in tqdm(dataset):
        collector.append({
            'segment_list': [{
                'str': item['ctx'], 
                'status': 'editing',
                'ground-truth': item['res'],
            }],
            'lang': 'zh',
            'topk': args['topk'],
        })
    print(f'[!] collect {len(collector)} samples for pipeline agent')
    return collector

def load_pipeline_data(path, size=1000):
    '''for pipeline and recall test'''
    data = read_text_data_utterances(path, lang='zh')
    dataset = []
    for i in range(0, len(data), 10):
        session = data[i:i+10]
        cache = []
        for label, utterances in session:
            if label == 1:
                cache.append(utterances[-1])
        # NOTE:
        dataset.append({
            'ctx': utterances[:-1],
            'res': cache
        })
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    collector = []
    for item in tqdm(dataset):
        collector.append({
            'segment_list': [{
                'str': item['ctx'], 
                'status': 'editing',
                'ground-truth': item['res'],
            }],
            'lang': 'zh',
            'topk': args['topk'],
        })
    print(f'[!] collect {len(collector)} samples for pipeline agent')
    return collector


def load_fake_partial_rerank_data(path, size=1000):
    # make sure the data reader
    if args['dataset'] in ['douban', 'ecommerce', 'ubuntu', 'lccc', 'lccc-large', 'restoration-200k']:
        dataset_ = read_text_data_utterances(path, lang='zh')
        # dataset = [(utterances[:-1], utterances[-1], None) for _, utterances in dataset]
        dataset = []
        for label, utterances in dataset_:
            if label == 1:
                ctx = ' '.join(utterances[:-1])
                num = random.choice([2, 3, 4])
                context = ctx[:-num]
                res = f'{ctx[-num:]} {utterances[-1]}'
                candidates = random.sample(utterances[:-1], 2)
                dataset.append((context, res, candidates))
    else:
        dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in dataset:
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'context': j[0], 
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

def load_fake_generation_data_from_writer_rank_corpus(path, size=1000):
    rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/test.rar'
    reader = torch.load(rar_path)
    reader.init_file_handler()
    dataset = [json.loads(reader.get_line(i)) for i in range(reader.size)]
    data = []
    for item in tqdm(dataset):
        sentences = [''.join(s.split()) for s in item['q']]
        # build the context and 1 ground-truth
        res_idx = random.randint(0 ,len(sentences)-1)
        length = len(sentences[res_idx])
        idx = random.randint(int(0.25*length), int(0.5*length))
        context = sentences[:res_idx] + [sentences[res_idx][:idx]]
        context = ''.join(context)
        data.append({
            'segment_list': [{
                'context': context, 
            }],
            'lang': 'zh',
        })
    data = random.sample(data, size)
    return data


def load_fake_rerank_data_from_writer_rank_corpus(path, size=1000):
    rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/test.rar'
    reader = torch.load(rar_path)
    reader.init_file_handler()
    dataset = [json.loads(reader.get_line(i)) for i in range(reader.size)]
    data = []
    for item in tqdm(dataset):
        sentences = [''.join(s.split()) for s in item['q']]
        # build the context and 1 ground-truth
        if random.random() < 0.5:
            res_idx = random.randint(1, len(sentences) - 1)
            context, response = sentences[:res_idx], sentences[res_idx]
        else:
            res_idx = random.randint(0 ,len(sentences)-1)
            length = len(sentences[res_idx])
            idx = random.randint(int(0.25*length), int(0.5*length))
            context = sentences[:res_idx] + [sentences[res_idx][:idx]]
            response = sentences[res_idx][idx:]
        # build 8 hard negative samples, add or delete tokens before the ground-truth response
        hard_neg = []
        for _ in range(9):
            if random.random() < 0.9:
                # add (copy)
                add_num = random.randint(3, 8)
                hard_neg.append(response[:add_num] + response)
            else:
                # delete
                delete_num = random.randint(3, 8)
                hard_neg.append(response[delete_num:])
        # build 10 random negative samples
        random_neg = []
        for _ in range(10):
            rr = random.choice(random.choice(dataset)['q'])
            rr = ''.join(rr.split())
            random_neg.append(rr)
        # overall candaidates
        # candidates = [response] + hard_neg + random_neg
        candidates = [response] + random_neg[:9]
        data.append({
            'segment_list': [{
                'context': context, 
                'candidates': candidates
            }],
            'lang': 'zh',
        })
    data = random.sample(data, size)
    return data


def load_fake_rerank_data(path, size=1000):
    # make sure the data reader
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

def load_wz_recall_data(path, size=1000):
    '''for pipeline and recall test'''
    dataset = read_text_data_line_by_line(path)
    dataset = set([line.strip().split('\t')[0] for line in dataset])
    data = []
    for i in tqdm(dataset):
        data.append({
            'segment_list': [{
                'str': i, 
                'status': 'editing'
            }],
            'lang': 'zh',
            'topk': args['topk'],
        })
    print(f'[!] collect {len(data)} samples for pipeline agent')
    return data

def load_fake_recall_data(path, size=1000):
    '''for pipeline and recall test'''
    if args['dataset'] in ['douban', 'ecommerce', 'ubuntu', 'lccc', 'lccc-large', 'restoration-200k']:
        # test set only use the context 
        dataset = read_text_data_utterances(path, lang='zh')
        dataset = [(utterances[:-1], utterances[-1], None) for label, utterances in dataset]
        dataset = [dataset[i] for i in range(0, len(dataset), 10)]
    elif args['dataset'] in ['potter']:
        dataset = read_text_data_utterances_potter(path, lang='zh')
        dataset = [(utterances, None) for utterances in dataset]
    elif args['dataset'] in ['poetry', 'novel_selected']:
        dataset = read_text_data_with_source(path, lang='zh')
    else:
        dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in tqdm(dataset):
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'str': j[0], 
                        'status': 'editing'
                    } for j in cache
                ],
                'lang': 'zh',
                'topk': args['topk'],
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    if cache:
        data.append({
            'segment_list': [
                {
                    'str': j[0], 
                    'status': 'editing'
                } for j in cache
            ],
            'lang': 'zh',
            'topk': args['topk'],
        })
    # data = random.sample(data, size)
    print(f'[!] collect {len(data)} samples for pipeline agent')
    return data

def SendPOST(url, port, method, params):
    '''
    import http.client

    parameters:
        1. url: 9.91.66.241
        2. port: 8095
        3. method:  /rerank or /recall
        4. params: json dumps string
    '''
    headers = {"Content-type": "application/json"}
    url = f'http://{url}:{port}{method}'
    data = requests.post(url, params)
    data = json.loads(data.text)
    return data

def test_recall_self_play(args):
    data = [
        {
            'segment_list': [
                {'str': '最近工作太累了', 'ground_truth': ''}
            ],'lang': 'zh'
        },
    ]
    # recall test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(range(50))
    data = data[0]
    vector = []
    for _ in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/recall', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        candidate = rest['item_list'][0]['candidates'][0]['text']
        vectors = rest['item_list'][0]['candidates'][0]['vectors']
        vector.append(np.array(vectors[0]))
        data = json.loads(data)
        data['segment_list'][0]['str'] = f'{data["segment_list"][0]["str"]} [SEP] {candidate}'
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
        pprint.pprint(rest)
        print(candidate)
        ipdb.set_trace()

    # draw
    return collections

def test_recall(args):
    # data = load_fake_recall_data(
    #     f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
    #     # f'{args["root_dir"]}/data/{args["dataset"]}/query.txt',
    #     size=args['size'],
    # )
    # data = load_wz_recall_data(
    #     f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
    #     size=args['size'],
    # )

    # data = [
    #     {
    #         'segment_list': [
    #             {'str': '今天要不要一起去看个电影呢'}
    #         ],'lang': 'zh'
    #     },
    #     {
    #         'segment_list': [
    #             {'str': '我女朋友终于要毕业啦'},
    #         ],'lang': 'zh'
    #     },
    #     {
    #         'segment_list': [
    #             {'str': '辛苦啦'},
    #         ],'lang': 'zh'
    #     },
    # ]
    def load_test_samples(path):
        with open(path) as f:
            lines = [line.strip() for line in f.readlines() if line[0] == '1']
            samples = []
            exist_set = set()
            for line in lines:
                items = line.split('\t')
                session = items[1:-2]
                session_string = ''.join(session)
                if session_string in exist_set:
                    continue
                else:
                    exist_set.add(session_string)
                ground_truth = items[-2]
                samples.append({
                    'segment_list': [
                        {'str': ' [SEP] '.join(session), 'ground_truth': ground_truth}    
                    ], 'lang': 'zh',
                })
        print(f'[!] load {len(samples)} sessions')
        return samples

    data = load_test_samples('/apdcephfs/share_916081/johntianlan/dialog-pretrain-ckpt/test.txt')

    # recall test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/recall', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
        pprint.pprint(rest)
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg recall time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_partial_rerank(args):
    data = load_fake_partial_rerank_data(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # rerank test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/rerank', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections


def test_rerank(args):
    # data = load_fake_rerank_data(
    #     f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
    #     size=args['size'],
    # )
    data = load_fake_rerank_data_from_writer_rank_corpus(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # rerank test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)

    # debug
    data= [
        {
            'segment_list': [{
                'context': '马东阳是一个帅哥', 
                'candidates': [
                    '，他是一个演员，他是一个歌手，他是一个演员，他是一个演员',
                    '，他是一个很有趣的人，他是一个很有魅力的人',
                    '，也是一个美女',
                    '。他的演技也是得到了大家的认可。他是一个很低调的演员',
                    '，也是一位演员，在《乡村爱情》里饰演的是刘能的父亲',
                    '，你觉得可能吗',
                    '，而且非常的有女人缘'
                ]
            }],
            'lang': 'zh',
        },
        {
            'segment_list': [{
                'context': '马东阳是一个帅哥', 
                'candidates': [
                    '，也是一个帅哥，也是一个帅哥，也是一个帅哥',
                    '，他在《天龙八部》中饰演的段誉是最经典的一个角色',
                    '，也是一个有魅力的男孩子，他有一个非常帅气的女朋友',
                    '，他是一个好人，他的好人品更是受到了很多人的喜欢，他是一个好人',
                    '，他也是一位演员，而且他也是一位很有才华的男演员',
                    '，更是大家心目中的学榜样'
                ]
            }],
            'lang': 'zh',
        },
        {
            'segment_list': [{
                'context': '今天天气风和日丽', 
                'candidates': [
                    '，非常适合去野外郊游',
                    '，马上就要下雨',
                    '，我准备和女朋友一起出门玩耍',
                    '，非常适合加班',
                    '，他是一个品学兼优的好学生',
                    '，但是我的心情非常的糟糕'
                ]
            }],
            'lang': 'zh',
        }
    ]

    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/rerank', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
        pprint.pprint(rest)
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_pipeline(args):
    data = load_pipeline_data_with_worker_id(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # pipeline test begin
    avg_times = []
    avg_recall_times = []
    avg_rerank_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(list(enumerate(data)))
    for idx, data in pbar:
        data = {'segment_list': [{'str': '想不想一起出去踢球'}]}
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/pipeline_evaluation', data)
        # rest = SendPOST(args['url'], args['port'], '/pipeline', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
            print(f'[!] ERROR happens in sample {idx}')
        else:
            collections.append(rest)
            # avg_times.append(rest['header']['core_time_cost_ms'])
            # avg_recall_times.append(rest['header']['recall_core_time_cost_ms'])
            # avg_rerank_times.append(rest['header']['rerank_core_time_cost_ms'])
        # pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
        # for debug
        print(rest)
    # show the result
    # for name in ['R@1000', 'R@500', 'R@100', 'R@50', 'MRR']:
    #     print(f'{name}: {rest["results"][name]}')
    avg_t = round(np.mean(avg_times), 4)
    avg_recall_t = round(np.mean(avg_recall_times), 4)
    avg_rerank_t = round(np.mean(avg_rerank_times), 4)
    print(f'[!] avg time cost: {avg_t} ms; avg recall time cost: {avg_recall_t} ms; avg rerank time cost {avg_rerank_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_generation_dialog(args):
    data = [{
        'segment_list': [
            {'context': ['几天跑了半个小时，真的太累了']},
            {'context': ['几天跑了半个小时，真的太累了', '那你现在在干嘛']},
            {'context': ['几天跑了半个小时，真的太累了', '那你现在在干嘛', '没干嘛']},
            {'context': ['刺猬很可爱！以前别人送了只没养，味儿太大！', '是很可爱但是非常臭', '是啊，没办法养', '那个怎么养哦不会扎手吗']},
            {'context': ['老铁家好吃贾三家不好吃']},
            {'context': ['话说红海真的好看！！！准备二刷了！！！但求多排场！！！', '我明天二刷']},
            {'context': ['为何突然伤感?']},
            {'context': ['杂糅太多反而摸不到主题了是吗']},
            {'context': ['你家真有钱', '从何说起？', '可以买粮']},
            {'context': ['火锅我在重庆成都吃了七八顿火锅', '哈哈哈哈！那我的嘴巴 可能要烂掉！']},
            {'context': ['为什么乡民总是欺负国高中生呢QQ', '如果以为选好科系就会变成比尔盖兹那不如退学吧']},
            {'context': ['京戏里头的人都是不自由的', '他们让人拿笼子给套起来了了']},
            {'context': ['你谈过恋爱么', '谈过，哎，别提了，伤心..']},
            {'context': ['前排，鲁迷们都起床了吧', '标题说助攻，但是看了那球，真是活生生的讽刺了']},
            {'context': ['看来你很爱钱', '噢是吗？那么你也差不多了']},
            {'context': ['看原版英文电影学纯正英语', '大爱老友记反复看了好多次 了', '一样光盘都快被我看花了', '那你现在的英语应该不错了']},
            {'context': ['这个会不会聚划算', '暂时没有哦', '后期会不会有', '不一定哦亲多多关注我们哦']},
            {'context': ['我今天腿都废了，你们过节，我搬砖', '辛苦啊，圣诞节还去赚大钱了加油', '毕竟是没男朋友的人，什么节都是一样的']},
            {'context': ['这居然是北京不是哈尔滨。']},
            {'context': ['运营商校园广告大PK，太欢乐了！哈哈哈。']},
            {'context': ['DelsaSposa糖果色系列婚纱，粉蓝红紫，俏皮又清新，你喜欢吗？']},
        ],
    }]

    pbar = tqdm(data)
    error_counter = 0
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/generation_dialog', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        pprint.pprint(rest)

def test_generation(args):
    data = load_fake_generation_data_from_writer_rank_corpus(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # rerank test begin
    avg_times = []
    collections = []
    error_counter = 0

    data = [{
        'segment_list': [
            {
                'context': '几天跑了半个小时，真的太累了',
            },
        ],
        'decoding_method': 'contrastive_batch_search',
        'beam_width': 5,
        'model_prediction_confidence': 0.6,
        'generation_num': 20,
        'max_gen_len': 64,
        # 'decoding_method': 'topk_topp_repetition_penalty_batch_fast_search',
    }]

    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/generation', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
        pprint.pprint(rest)
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_copygeneration(args):
    args['mode'] = 'test'
    args['model'] = 'copygeneration'
    config = load_config(args)
    args.update(config)
    test_data, test_iter, _ = load_dataset(args)
    pbar = tqdm(test_iter)
    for batch in pbar:
        data = {
            'prefix': batch['prefix'],
            'ground_truth': batch['ground_truth'],
            'decoding_method': 'retrieval-generation-search',
            'generation_method': 'nucleus-search',
            'beam_width': 5,
            'model_prediction_confidence': 0.9,
            'generation_num': 20,
            'max_gen_len': 64,
            'recall_topk': 20,
            'topk': 8,
            'topp': 0.93,
            'lang': 'en'
        }
        ipdb.set_trace()
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/copygeneration', data)
        pprint.pprint(rest)
    return collections

def test_evaluation(args):
    data = {
        'segment_list': [
            {
                'context': ['今天要不要一起去踢个球放松一下'],
                'candidate1': '好啊好啊',
                'candidate2': '我今天就不去了，下次约'
            },
            {
                'context': ['今天要不要一起去踢个球放松一下'],
                'candidate1': '这道题怎么做啊',
                'candidate2': '我今天就不去了，下次约'
            },    
            {
                'context': ['今天要不要一起去踢个球放松一下', '要不今天就算了'],
                'candidate1': '今天为什么不行啊',
                'candidate2': '明天可不可以啊'
            },
            {
                'context': ['你到底为什么要和我在一起'],
                'candidate1': '为了更好的照顾你',
                'candidate2': '你是谁啊，你配吗'
            },
            {
                'context': ['你到底为什么要和我在一起'],
                'candidate1': '为了更好的照顾你',
                'candidate2': '为了更好的照顾你',
            },
            {
                'context': ['你到底为什么要和我在一起'],
                'candidate1': '你到底为什么要和我在一起',
                'candidate2': '为了更好的照顾你',
            },
        ]        
    }
    data = json.dumps(data)
    rest = SendPOST(args['url'], args['port'], '/evaluation', data)
    pprint.pprint(rest)
    ipdb.set_trace()

if __name__ == '__main__':
    # topk rewrite
    args = vars(parser_args())

    # set the random seed
    random.seed(args['seed'])

    args['root_dir'] = '/apdcephfs/share_916081/johntianlan/MyReDial'
    MAP = {
        'recall': test_recall,
        'recall_self_play': test_recall_self_play,
        'rerank': test_rerank,
        'partial_rerank': test_partial_rerank,
        'pipeline': test_pipeline,
        'generation': test_generation,
        'generation_dialog': test_generation_dialog,
        'copygeneration': test_copygeneration,
        'evaluation': test_evaluation,
    }
    collections = MAP[args['mode']](args)
    
    # write into log file
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/test_api_{args["mode"]}_{args["port"]}_{args["prefix_name"]}_{args["worker_id"]}_log.txt'
    with open(write_path, 'w') as f:
        for sample in tqdm(collections):
            data = sample['item_list']
            if sample['header']['ret_code'] == 'fail':
                continue
            if args['mode'] == 'pipeline':
                for item in data:
                    string = '\t'.join(item['context'])
                    f.write(f'[Context ] {string}\n')
                    f.write(f'[Response] {item["response"]}\n\n')
                    # f.write(f'[MRR Metric] {item["mrr"]}\n\n')
            elif args['mode'] == 'recall':
                for item in data:
                    f.write(f'[Context] {item["context"]}\n')
                    for idx, neg in enumerate(item['candidates']):
                        f.write(f'[Cands-{idx}] {neg["text"]}\n')
                    f.write('\n')
            elif args['mode'] in ['rerank', 'partial_rerank']:
                for item in data:
                    f.write(f'[Context] {item["context"]}\n')
                    for i in item['candidates']:
                        f.write(f'[Score {round(i["score"], 2)}] {i["str"]}\n')
                    f.write('\n')
            elif args['mode'] in ['generation']:
                for item in data:
                    f.write(f'[Context] {item["context"]}\n')
                    for i in item['generations']:
                        f.write(f'[Candidates] {i["str"]}\n')
                    f.write('\n')
            else:
                raise Exception(f'[!] Unkown mode: {args["mode"]}')

    print(f'[!] write the log into file: {write_path}')
