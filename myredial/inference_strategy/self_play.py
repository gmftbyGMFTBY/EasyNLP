from inference import *
from header import *
from dataloader.utils import *
from config import *
from .utils import *

'''
self-play strategy to generate the additional data samples for training

Make sure the deploy config is set in the config/base.yaml
'''

def remove_duplicate_punctuation(utterance):
    chars = []
    punctuations = ['。', '.', '！', '#', '~', '～', '?', '!', '？', '·', ' ', '）', '（', '(', ')', '{', '}']
    for i in utterance:
        if i in punctuations:
            if len(chars) > 0 and i == chars[-1]:
                continue
        chars.append(i)
    return ''.join(chars)

def load_utterances(args, path):
    utterances = read_response_data_full(path, lang=args['lang'], turn_length=5)
    utterances = list(set(utterances))
    interval = len(utterances) // dist.get_world_size()
    chunks = []
    for i in range(0, len(utterances), interval):
        chunks.append(utterances[i:i+interval])
    chunks = chunks[:dist.get_world_size()]
    assert len(chunks) == dist.get_world_size()
    utterances = chunks[args['local_rank']]
    print(f'[!] collect {len(utterances)} for process: {args["local_rank"]}')
    return utterances

def load_agent(args):
    recall_args = load_deploy_config('recall')
    recall_args['dataset'] = args['dataset']
    recall_args['model'] = args['model']
    recallagent = RecallAgent(recall_args)
    print(f'[!] load the recall agents over')
    return recallagent

def self_play_strategy(args):
    # set the seed
    random.seed(args['seed'])

    # load the context embeddings of the extra data samples
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    utterances = load_utterances(args, path)
    random.shuffle(utterances)

    recallagent = load_agent(args)

    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(device=args['local_rank'])
    print(f'[!] read the faiss index over, begin to search from the index')

    # self-play
    dataset, idx = [], 0
    topk = args['gen_dataset_topk']
    batch_size = args['batch_size']
    with tqdm(total=args['gen_dataset_num']) as pbar:
        while len(dataset) < args['gen_dataset_num']:
            # start utterances
            st_us = utterances[idx:idx+batch_size]
            beam = [[remove_duplicate_punctuation(i)] for i in st_us]
            for _ in range(args["gen_dataset_ctx_length"]):
                # pack up for recall
                recall_inpt = [{'str': session} for session in beam]
                candidates = recallagent.work(recall_inpt, topk=args['gen_dataset_topk'])
                counter, snr = 0, []
                valid = [True for _ in range(len(beam))]
                for candidate in candidates:
                    candidate = [remove_duplicate_punctuation(i['text']) for i in candidate]
                    # remove the duplicate utterances that appears in the conversation history
                    candidate = list(set(candidate) - set(beam[counter]))
                    if len(candidate) == 0:
                        valid[counter] = False
                    else:
                        r = candidate[0]
                        # snr
                        candidate.remove(r)
                        snr.append(candidate[:20])
                        # beam append
                        beam[counter].append(r)
                    counter += 1
                beam = [b for v, b in zip(valid, beam) if v]
                if len(beam) == 0:
                    break
            beam = [{'q': us[:-1], 'r': us[-1], 'snr': snr_} for us, snr_ in zip(beam, snr)]
            dataset.extend(beam)
            idx += batch_size
            pbar.update(len(beam))
            pbar.set_description(f'[!] collect {len(dataset)} self-play samples')
            if len(dataset) >= args['gen_dataset_num']:
                break
            if idx >= len(utterances):
                break

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gen_{args["local_rank"]}.txt'
    with open(path, 'w') as f:
        for item in tqdm(dataset):
            string = json.dumps(item)
            f.write(f'{string}\n')
    print(f'[!] self-play generate {len(dataset)} samples, save into {path}')
