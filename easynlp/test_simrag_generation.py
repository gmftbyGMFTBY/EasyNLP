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

    searcher_args['model'] = 'dual-bert'
    config = load_config(searcher_args)
    searcher_args.update(config)

    simcse_args['model'] = 'simcse'
    simcse_args['mode'] = 'test'
    config = load_config(simcse_args)
    simcse_args.update(config)

    searcher_args_['model'] = 'dual-bert'
    searcher_args_['mode'] = 'inference'
    config = load_config(searcher_args_)
    searcher_args_.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)

    # load simcse scorer
    simcse_agent = load_model(simcse_args)
    pretrained_model_name = simcse_args['pretrained_model'].replace('/', '_')
    save_path = f'{simcse_args["root_dir"]}/ckpt/{simcse_args["dataset"]}/{simcse_args["model"]}/best_{pretrained_model_name}_{simcse_args["version"]}.pt'
    simcse_agent.load_model(save_path)
    print(f'[!] init the simcse scorer over')

    # load the dialogue genenerator
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    print(f'[!] init the generator over')

    # searcher
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

    collection = []
    f = open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_simrag.txt', 'w')
    beta = agent.model.args['beta']
    copy_token_num = agent.model.args['copy_token_num']
    for batch in tqdm(test_iter):
        context_list = batch['context_list']
        ground_truth = batch['ground_truth']
        batch = searcher_agent.inference_context_one_sample(context_list)
        retrieval_list = searcher._search(batch, topk=args['recall_topk'])[0]
        agent.model.args['beta'] = beta
        # res_with_beta = agent.simrag_talk(context_list, retrieval_list=retrieval_list, scorer=simcse_agent, copy_token_num=copy_token_num)
        agent.model.args['beta'] = 0.
        res = agent.simrag_talk(context_list, retrieval_list=retrieval_list, scorer=simcse_agent, copy_token_num=copy_token_num)
        topk_topp_res1 = agent.simrag_talk(context_list, topk_topp=True, retrieval_list=retrieval_list, copy_token_num=copy_token_num)
        topk_topp_res2 = agent.simrag_talk(context_list, topk_topp=True, retrieval_list=retrieval_list, copy_token_num=copy_token_num)
        topk_topp_res3 = agent.simrag_talk(context_list, topk_topp=True, retrieval_list=retrieval_list, copy_token_num=copy_token_num)
        item = {
            'context': context_list,
            'ground_truth': ground_truth,
            'retrieval_list': retrieval_list,
            # 'rag_response': res_with_beta,
            'cl_response': res,
            'topk_topp_response1': topk_topp_res1,
            'topk_topp_response2': topk_topp_res2,
            'topk_topp_response3': topk_topp_res3,
        }
        f.write(f'[Context     ] {item["context"]}\n')
        f.write(f'[Retrieval   ] {item["retrieval_list"]}\n')
        f.write(f'[Ground-Truth] {item["ground_truth"]}\n')
        # f.write(f'[SimRAG Res  ] {item["rag_response"]}\n')
        f.write(f'[CL Res      ] {item["cl_response"]}\n')
        f.write(f'[Topk-p Res  ] {item["topk_topp_response1"]}\n')
        f.write(f'[Topk-p Res  ] {item["topk_topp_response2"]}\n')
        f.write(f'[Topk-p Res  ] {item["topk_topp_response3"]}\n\n')
        f.flush()


if __name__ == "__main__":
    args = vars(parser_args())
    main_generation(**args)
