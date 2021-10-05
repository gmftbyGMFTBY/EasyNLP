from inference import *
from header import *
from .utils import *

'''
gray strategy generates the hard negative samples (gray samples) for each conversation context in the training and testing dataset:

Need the BERTDualInferenceFullContextDataset'''


def gray_one2many_strategy(args):
    # collect the gray negative dataset
    embds, contexts, responses = [], [], []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, context, response = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}_{idx}.pt'        
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
    lossing = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result, distance = searcher._search_dis(batch, topk=args['pool_size'])
        for c, r, rest, dis in zip(context, response, result, distance):
            ipdb.set_trace()
            # ignore the invalid results in the faiss searching results
            rest = [i for i, j in zip(rest, dis) if j < 1e8]
            rest = remove_duplicate_and_hold_the_order(rest)
            # remove the candidate that in the conversation context
            rest = [u for u in rest if u not in c]
            # remove the ground-truth
            if r in rest:
                rest.remove(r)
            if len(rest) < args['gray_topk']:
                lossing += 1
                continue
            rest = rest[:args['gray_topk']]
            collection.append({'q': c, 'r': r, 'snr': rest})
        pbar.set_description(f'[!] found {lossing} error samples')
    print(f'[!] lossing {lossing} samples that are invalid')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_dpr_gray.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')

