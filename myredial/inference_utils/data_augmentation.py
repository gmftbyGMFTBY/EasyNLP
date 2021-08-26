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

    # full split
    n_ctx, n_res, n_ret = [], [], []
    for c, r, re in zip(contexts, responses, results):
        utterances = c + [r]
        start_num = max(1, len(utterances) - args['full_turn_length'])
        for i in range(start_num, len(utterances)):
            n_ctx.append(utterances[:i])
            n_res.append(utterances[i])
            n_ret.append(re)
    torch.save([n_ctx, n_res, n_ret], path)
    print(f'[!] save the data into {path}')

