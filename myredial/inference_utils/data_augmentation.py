from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index
'''

def da_strategy(args):
    contexts, responses, results = [], [], []
    for i in tqdm(range(args['nums'])):
        c, r, re = torch.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/inference_bert_mask_da_{i}.pt'
        )
        print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_bert_mask_da_{i}.pt')
        res = []
        for re_ in re:
            re_ = [ii.strip() for ii in re_ if ii.strip()]
            res.append(re_)
        contexts.extend(c)
        responses.extend(r)
        results.extend(res)
    print(f'[!] collect {len(contexts)} samples')
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bert_mask_da_results.pt'
    torch.save([contexts, responses, results], path)
    print(f'[!] save the data into {path}')

