from inference import *
from header import *

'''
unparallel strategy generates the pesudo positive pair given the built index
1. context-for-response
2. search the pesudo label
'''

def unparallel_strategy(args):
    # load the context embeddings of the extra data samples
    embds, responses = [], []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, response = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_for_response_{args["model"]}_{i}_{idx}.pt'        
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_context_for_response_{args["model"]}_{i}_{idx}.pt')
            except:
                break
            embds.append(embd)
            responses.extend(response)
    embds = np.concatenate(embds) 
    contexts = responses

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
        result = searcher._search(batch, topk=args['pool_size'])
        for c, rest in zip(context, result):
            if c in rest:
                rest.remove(c)
            snr = rest[:args['topk']]
            collection.append({'q': c, 'snr': snr})

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_unparallel.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')
