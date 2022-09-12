from inference import *
from model import *
from header import *
from .utils import *
from es.es_utils import *


def gray_hard_strategy(args):
    embds, contexts, responses = [], [], []
    for idx in range(100):
        try:
            embd, context, response = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{args["local_rank"]}_{idx}.pt'        
            )
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        except:
            break
    embds = np.concatenate(embds) 
    print(f'[!] load {len(contexts)} contexts for generating the gray candidates')
    
    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = []
    bad_response_num = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    sample_num = 0
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['gray_topk'])
        for c, r, rest in zip(context, response, result):
            rest = list(set([u for u in rest if u not in c]))
            if r in rest:
                rest.remove(r)
            collection.append({'q': c, 'r': r, 'hp': rest})
            ipdb.set_trace()
        sample_num += len(batch)
        pbar.set_description(f'[!] total response: {sample_num}')
    print(f'[!] total samples: {len(embds)}; bad response num: {bad_response_num}')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_{args["local_rank"]}.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')

def gray_hard_test_strategy(args):
    embds, contexts, responses = [], [], []
    for idx in range(100):
        try:
            embd, context, response = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_test_context_{args["model"]}_{args["local_rank"]}_{idx}.pt'        
            )
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        except:
            break
    embds = np.concatenate(embds) 
    print(f'[!] load {len(contexts)} contexts for generating the gray candidates')
    
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
    collection = []
    bad_response_num = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    sample_num = 0
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['gray_topk'])
        for c, r, rest in zip(context, response, result):
            rest = list(set([u for u in rest if u not in c]))
            if r in rest:
                rest.remove(r)
            collection.append({'q': c, 'r': r, 'hp': rest})
        sample_num += len(batch)
        pbar.set_description(f'[!] total response: {sample_num}')
    print(f'[!] total samples: {len(embds)}; bad response num: {bad_response_num}')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_{args["local_rank"]}.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')
