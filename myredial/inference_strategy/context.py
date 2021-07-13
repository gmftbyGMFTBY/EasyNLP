from inference import *
from header import *

'''
context strategy inference the context sentence and save it into the faiss index,
the inferenced embedding will be used for the gray strategy mode,
'''

def context_strategy(args):
    # inference the context and do the q-q matching
    embds, corpus = [], []
    for i in tqdm(range(args['nums'])):
        embd, context, response = torch.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}.pt'        
        )
        embds.append(embd)
        corpus.extend([(c, r) for c, r in zip(context, response)])
    embds = np.concatenate(embds) 
    # write into faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], q_q=True)
    searcher._build(embds, corpus)
    searcher.save(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_corpus.ckpt',
    )
    print(f'[!] save the context faiss over')
