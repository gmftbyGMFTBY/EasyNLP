from inference import *
from header import *

'''
gray_one2many strategy generates the hard negative samples (gray samples) for each conversation context in the training and testing dataset, and try to find the potential positive samples for training'''

def load_bert_fp_agent(args):
    args['model'] = 'bert-fp-original'
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    return agent


def gray_one2many_strategy(args):
    # load bert-fp agent
    bert_fp_agent = load_bert_fp_agent(deepcopy(args))
    print(f'[!] load bert-fp agent over')

    # read the embeddings of the conversation context
    embds, contexts, responses = torch.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{args["local_rank"]}.pt'
    )
    print(f'[!] {args["local_rank"]} load context size: {len(embds)}')
    assert len(embds) == len(contexts) and len(contexts) == len(responses)

    # read the candidate faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(args['local_rank'])
    print(f'[!] read the faiss index over, begin to search from the index')

    # search and re-label (pesudo label)
    collection = []
    for i in tqdm(range(0, len(embds), args['batch_size'])):
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['pool_size'])
        packages = []
        for c, r, rest in zip(context, response, result):
            if r in rest:
                rest.remove(r)
            responses_ = [r] + rest[:args['topk']]
            # label by bert-fp
            packages.append({
                'context': c,
                'candidates': responses_,
            })
        scores_list = bert_fp_agent.rerank(packages)
        for score, package in zip(scores_list, packages):
            nr = []
            for idx in range(1, len(score)):
                if score[idx] >= score[0]:
                    nr.append(package['candidates'][idx])
            collection.append({'q': package['context'], 'r': package['candidates'][0], 'pnr': nr})

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_one2many_local_rank_{args["local_rank"]}.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')
