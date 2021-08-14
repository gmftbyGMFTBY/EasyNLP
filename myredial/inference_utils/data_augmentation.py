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
        contexts.extend(c)
        responses.extend(r)
        results.extend(re)
    print(f'[!] collect {len(contexts)} samples')
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bert_mask_da_results.pt'
    torch.save([contexts, responses, results], path)

