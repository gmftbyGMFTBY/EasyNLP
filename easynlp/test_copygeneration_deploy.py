from header import *
from dataloader import *
from model import *
from config import *
from inference import *
from es import *
from flask import Flask, request, jsonify, make_response, session

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    parser.add_argument('--port', type=int, default=22330)
    return parser.parse_args()

def load_base_data():
    if args['dataset'] == 'wikitext103':
        data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/'
        num = 8
        nlp = spacy.load('en_core_web_sm')
    else:
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        # num = 32
        
        # domain adaption dataset
        data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_zh_news'
        num = 8
    base_data = {}
    for i in tqdm(range(num)):
        file = os.path.join(data_path, f'searched_results_{i}_base.txt')
        with open(file) as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                base_data[line['index']] = line['results']
    print(f'[!] collect {len(base_data)} documents')
    return base_data

def create_app(**args):
    app = Flask(__name__)

    # init the model
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

    # test_data, test_iter, _ = load_dataset(args)
    base_data = load_base_data()

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

    agent.model.init_searcher(searcher_agent, searcher, base_data)

    # faiss searcher
    # faiss_searcher = Searcher(
    #     faiss_searcher_args['index_type'] ,
    #     dimension=faiss_searcher_args['dimension'],
    #     nprobe=faiss_searcher_args['index_nprobe']
    # )
    # pretrained_model_name = faiss_searcher_args['pretrained_model'].replace('/', '_')
    # model_name = faiss_searcher_args['model']
    # faiss_searcher.load(
    #     f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
    #     f'{faiss_searcher_args["root_dir"]}/data/{faiss_searcher_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    # )
    # agent.model.init_faiss_searcher(faiss_searcher)
    # print(f'[!] init model over')

    @app.route('/copygeneration', methods=['POST'])
    def generation_api():
        try:
            data = json.loads(request.data)
            batch = {}
            batch['prefix'] = data['prefix']
            batch['ground_truth'] = data['ground_truth']

            batch['decoding_method'] = 'contrastive-search' if 'decoding_method' not in data else data['decoding_method']
            batch['generation_method'] = 'contrastive-search' if 'generation_method' not in data else data['generation_method']
            batch['topk'] = 8 if 'topk' not in data else data['topk']
            batch['topp'] = 0.93 if 'topk' not in data else data['topp']
            batch['beam_width'] = 5 if 'beam_width' not in data else data['beam_width']
            batch['model_prediction_confidence'] = 0.4 if 'model_prediction_confidence' not in data else data['model_prediction_confidence']
            batch['phrase_alpha'] = 1. if 'phrase_alpha' not in data else data['phrase_alpha']
            batch['update_step'] = 64 if 'update_step' not in data else data['update_step']
            batch['max_gen_len'] = data['max_gen_len'] if 'max_gen_len' in data else 32
            batch['softmax_temp'] = data['softmax_temp'] if 'softmax_temp' in data else 0.001
            
            if 'recall_topk' in data:
                agent.model.args['recall_topk'] = data['recall_topk']
            res = agent.model.work(batch) 
            succ = True
        except Exception as error:
            print(error)
            succ = False
            res = None
        result = {
            'result': res,
            'prefix': data['prefix'],
            'ground-truth': data['ground_truth'],
            'succ': succ
        }
        return jsonify(result)

    return app

if __name__ == "__main__":
    args = vars(parser_args())
    app = create_app(**args)
    app.run(host='0.0.0.0', port=args['port'])
