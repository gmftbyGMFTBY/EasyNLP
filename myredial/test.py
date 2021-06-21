from header import *
from dataloader import *
from model import *
from config import *
from inference import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--recall', dest='recall', action='store_true')
    parser.add_argument('--no-recall', dest='recall', action='store_false')
    return parser.parse_args()


def main(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    print('test', args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    # (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model(test_iter, print_output=True)
    outputs = agent.test_model(test_iter, print_output=True)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)

def main_recall(**args):
    '''test the recall with the faiss index'''
    # use test mode args load test dataset and model
    inf_args = deepcopy(args)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    print('test', args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    test_data, test_iter, _ = load_dataset(args)

    # ===== use inference args ===== #
    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)
    print(f'inference', inf_args)

    # load faiss index
    searcher = Searcher(inf_args['index_type'], dimension=inf_args['dimension'])
    model_name = inf_args['model']
    pretrained_model_name = inf_args['pretrained_model']
    searcher.load(
        f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',        
        f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',        
    )
    print(f'[!] load faiss over')

    # test recall (Top-20, Top-100)
    pbar = tqdm(test_iter)
    counter, acc = 0, 0
    cost_time = []
    for batch in pbar:
        ids = batch['ids'].unsqueeze(0)
        ids_mask = torch.ones_like(ids)
        vector = agent.model.get_ctx(ids, ids_mask)   # [E]
        if not searcher.binary:
            vector = vector.cpu().numpy()

        bt = time.time()
        rest = searcher._search(vector, topk=inf_args['topk'])[0]
        et = time.time()
        cost_time.append(et - bt)

        gt_candidate = batch['text']
        if len(gt_candidate) == 0:
            continue
        for text in gt_candidate:
            if text in rest:
                acc += 1
                break
        counter += 1
    topk_metric = round(acc/counter, 4)
    avg_time = round(np.mean(cost_time)*1000, 2)    # ms
    print(f'[!] Top-{inf_args["topk"]}: {topk_metric}')
    print(f'[!] Average Times: {avg_time} ms')
    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_recall_{pretrained_model_name}.txt', 'w') as f:
        print(f'Top-{inf_args["topk"]}: {topk_metric}', file=f)
        print(f'Average Times: {avg_time} ms', file=f)

if __name__ == "__main__":
    args = vars(parser_args())
    if args['recall']:
        print(f'[!] Make sure that the inference script of model({args["model"]}) on dataset({args["dataset"]}) has been done.')
        main_recall(**args)
    else:
        main(**args)
