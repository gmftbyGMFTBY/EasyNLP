from inference import *
from model import *
from header import *
from .utils import *
from es.es_utils import *
from dataloader import *


def inference_time_cost_strategy(args, agent):
    # load the textual dataset and process them
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test.txt'
    dataset_ = read_text_data_utterances(path, lang=args['lang'])
    dataset = []
    for i in range(0, len(dataset_), 10):
        line = dataset_[i][1][:-1]
        dataset.append(line)
    print(f'[!] collect {len(dataset)} samples from {path}')

    length = np.mean([len(''.join(i)) for i in dataset])
    print(f'[!] the average length of the dataset context is: {round(length, 4)}')

    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    # searcher.move_to_gpu(device=args['local_rank'])

    # search
    pbar = tqdm(range(0, len(dataset), args['batch_size']))
    counter = 0
    qr_time_cost, cr_time_cost = 0, 0
    for i in pbar:
        context = dataset[i:i+args['batch_size']]
        # 1. query representation
        batch, t = agent.inference_context_one_batch(context)
        qr_time_cost += t
        # 2. candidate recall
        bt = time.time()
        result = searcher._search(batch, topk=args['gray_topk'])
        cr_time_cost += time.time() - bt
        counter += 1
    print(f'[!] qr_time_cost: {round(qr_time_cost/counter*1000, 4)}')
    print(f'[!] cr_time_cost: {round(cr_time_cost/counter*1000, 4)}')
