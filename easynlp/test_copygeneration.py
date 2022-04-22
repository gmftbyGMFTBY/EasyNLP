from header import *
from dataloader import *
from model import *
from config import *
from inference import *
from es import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def main_generation(**args):
    args['mode'] = 'test'
    searcher_args = deepcopy(args)
    config = load_config(args)
    args.update(config)
    
    faiss_searcher_args = deepcopy(args)
    faiss_searcher_args['model'] = 'phrase-copy'
    config = load_config(faiss_searcher_args)
    faiss_searcher_args.update(config)

    searcher_args['model'] = 'simcse'
    config = load_config(searcher_args)
    searcher_args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)

    # load the dialogue genenerator
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    print(f'[!] init the copygeneration over')

    # searcher
    searcher = Searcher(
        searcher_args['index_type'],
        dimension=searcher_args['dimension'],
        nprobe=searcher_args['index_nprobe']
    )
    pretrained_model_name = searcher_args['pretrained_model'].replace('/', '_')
    model_name = searcher_args['model']
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    )
    searcher_args['local_rank'] = 0
    searcher_agent = load_model(searcher_args)
    save_path = f'{args["root_dir"]}/ckpt/{searcher_args["dataset"]}/{searcher_args["model"]}/best_{pretrained_model_name}_{searcher_args["version"]}.pt'
    searcher_agent.load_model(save_path)
    print(f'[!] init the searcher and dual-bert model over')

    agent.model.init_searcher(searcher_agent, searcher, test_data.base_data)

    # faiss searcher
    faiss_searcher = Searcher(
        faiss_searcher_args['index_type'] ,
        dimension=faiss_searcher_args['dimension'],
        nprobe=faiss_searcher_args['index_nprobe']
    )
    pretrained_model_name = faiss_searcher_args['pretrained_model'].replace('/', '_')
    model_name = faiss_searcher_args['model']
    faiss_searcher.load(
        f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    )
    agent.model.init_faiss_searcher(faiss_searcher)
    print(f'[!] init model over')

    collection = []
    f = open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_copygeneration.txt', 'w')
    for batch in tqdm(test_iter):
        # parameters of decoding strategies
        batch['topk'] = 8
        batch['topp'] = 0.93
        batch['beam_width'] = 5
        batch['model_prediction_confidence'] = 0.4
        batch['phrase_alpha'] = 1.
        # batch['generation_method'] = 'topk-topp-search'
        batch['generation_method'] = 'contrastive-search'
        batch['update_step'] = 64

        if not batch['prefix']:
            continue
        
        for decoding_method in [
            'topk-topp-search', 
            # 'greedy-search', 
            # 'beam-search', 
            'contrastive-search', 
            # 'retrieval-search', 
            # 'retrieval-generation-search'
            'retrieval-generation-search-e2e'
        ]:
            batch['decoding_method'] = decoding_method
            res = agent.model.work(batch) 
            batch[f'{decoding_method}_response'] = res
        string = json.dumps(batch, ensure_ascii=False)
        f.write(string + '\n')
        f.flush()

if __name__ == "__main__":
    args = vars(parser_args())
    main_generation(**args)
