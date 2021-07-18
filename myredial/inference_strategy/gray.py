from inference import *
from header import *

'''
gray strategy generates the hard negative samples (gray samples) for each conversation context in the training and testing dataset'''

def gray_strategy(args):
    # collect the gray negative dataset
    embds, contexts, responses = [], [], []
    for i in tqdm(range(args['nums'])):
        embd, context, response = torch.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}.pt'        
        )
        embds.append(embd)
        contexts.extend(context)
        responses.extend(response)
    embds = np.concatenate(embds) 

    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu()
    print(f'[!] read the faiss index over, begin to search from the index')

    # search
    # NOTE: Make sure the responses are saved in the faiss index
    collection = []
    for i in tqdm(range(0, len(embds), args['batch_size'])):
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['pool_size'])
        for c, r, rest in zip(context, response, result):
            if r in rest:
                rest.remove(r)
            # nr = random.sample(rest, args['topk'])
            nr = rest[-args['topk']:]
            super_nr = rest[:args['topk']]
            collection.append({'q': c, 'r': r, 'nr': nr, 'snr': super_nr})

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')
