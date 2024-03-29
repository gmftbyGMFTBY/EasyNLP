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
    searcher_args_ = deepcopy(args)
    simcse_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

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
    print(f'[!] init the generator over')

    # simsce searcher
    searcher = Searcher(
        searcher_args_['index_type'] ,
        dimension=searcher_args_['dimension'],
        nprobe=searcher_args_['index_nprobe']
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

    # generation
    collection = []
    f = open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_copygeneration.txt', 'w')
    for batch in tqdm(test_iter):
        context_list = batch['context_list']
        ground_truth = batch['ground_truth']
        batch = searcher_agent.inference_context_one_sample(context_list)
        retrieval_list = searcher._search(batch, topk=args['recall_topk'])[0]
        res = agent.copygenerate(context_list, retrieval_list=retrieval_list, scorer=simcse_agent, copy_token_num=copy_token_num)
        f.write(f'[Context     ] {context}\n')
        f.write(f'[Ground-Truth] {ground_truth}\n')
        f.write(f'[CL Res      ] {res}\n')
        f.flush()


if __name__ == "__main__":
    args = vars(parser_args())
    main_generation(**args)
