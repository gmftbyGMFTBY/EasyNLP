from inference import *
from model import *
from header import *
from .utils import *
from es.es_utils import *


def gray_rag_bert_ft_strategy(args):

    embds, texts = [], []
    for i in range(32):
        for idx in range(100):
            try:
                path = f'{args["root_dir"]}/data/{args["dataset"]}/inference_dialog_context_dual-bert_{i}_{idx}.pt'
                embd, text = torch.load(path)
                embds.append(embd)
                texts.extend(text)
                print(f'[!] load file {path} over')
            except:
                break
    embds = np.concatenate(embds) 
    print(f'[!] load {len(texts)} contexts for generating the gray candidates')

    with open(f'{args["root_dir"]}/data/{args["dataset"]}/train_cross_domain.txt') as f:
        dataset = {}
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            dataset[line[0]] = (line[1:-2], line[-1], line[-2])
    
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
    print(f'[!]')

    # search
    collection = []
    bad_response_num = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    sample_num = 0
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        index = texts[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['gray_topk']+1)
        for idx, rest in zip(index, result):
            rest = list(set([u for u in rest if u != idx]))
            collection.append({
                'conversation': idx,
                'gray_conversations': rest
            })
        ipdb.set_trace()
        print(f'[!]', dataset[idx])
        for ii in rest:
            print(f'[!]', dataset[ii])
        sample_num += len(batch)
        pbar.set_description(f'[!] total response: {sample_num}')
    print(f'[!] total samples: {len(embds)}; bad response num: {bad_response_num}')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_rag_bert_ft.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item, ensure_ascii=False)
            f.write(f'{string}\n')
