from inference import *
from header import *
from .utils import *


def gray_simcse_strategy(args):
    # read the embeddings
    embds, texts, indexes = [], [], []
    base = 24
    for idx in range(100):
        try:
            embd, index = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{args["local_rank"]+base}_{idx}.pt'
            )
            print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{args["local_rank"]+base}_{idx}.pt')
        except Exception as error:
            break
        embds.append(embd)
        indexes.extend(index)
    embds = np.concatenate(embds)
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
    collection = {}
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        index = indexes[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['pool_size'])
        for idx, rest in zip(index, result):
            if idx in rest:
                rest.remove(idx)
            if idx in collection:
                collection[idx].append({
                    'index': idx,
                    'cands': rest
                })
            else:
                collection[idx] = [{'index': idx, 'cands': rest}]
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse_{args["local_rank"]+base}.pt'
    torch.save(collection, path)
