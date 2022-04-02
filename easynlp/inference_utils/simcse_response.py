from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index
'''

def simcse_response_strategy(args):
    embds, texts = [], []
    text_ids = []
    already_added = []
    current_num = 0
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, text, text_id = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt')
                current_num += len(embd)
            except:
                break
            embds.append(embd)
            texts.extend(text)
            text_ids.extend(text_id)
            already_added.append((i, idx))
            print(f'[!] collect embeddings: {current_num}')
            if current_num > 2000000:
                break
        if current_num > 2000000:
            break
    embds = np.concatenate(embds) 
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher._build(embds, text_ids, speedup=True)
    print(f'[!] train the searcher over')

    # add the external dataset
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            if (i, idx) in already_added:
                continue
            try:
                embd, text, text_id = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{i}_{idx}.pt')
            except:
                break
            searcher.add(embd, text_id)
    print(f'[!] total samples: {searcher.searcher.ntotal}')

    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher.save(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    print(f'[!] save faiss index over')
