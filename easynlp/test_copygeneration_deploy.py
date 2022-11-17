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
    parser.add_argument('--partial', type=float, default=1.0)
    return parser.parse_args()

def load_base_data(dataset):
    if dataset in ['wikitext103', 'copygeneration_lawmt']:
        nlp = spacy.load('en_core_web_sm')
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/backup_v4_data'
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103'
        data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_wikitext103/backup_v2_data'
        
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_lawmt/'
        num = 8
        
        
        
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_en_wiki/backup_v4_data'
        # num = 32
    elif dataset in ['en_wiki']:
        # en-wiki test set with en-wiki larger memory
        nlp = spacy.load('en_core_web_sm')
        data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_en_wiki/backup_v4_data'
        num = 32
    elif dataset in ['copygeneration', 'copygeneration_zh_news']:
        data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data/backup_v2_data/'
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_data'
        num = 32
        # domain adaption dataset
        # data_path = f'/apdcephfs/share_916081/johntianlan/copygeneration_zh_news'
        # num = 8
    base_data = {}
    for i in tqdm(range(num)):
        file = os.path.join(data_path, f'searched_results_{i}_base.txt')
        try:
            with open(file) as f:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)
                    base_data[line['index']] = line['results']
        except:
            print(f'[!] load file failed: {file}')
            continue
    print(f'[!] collect {len(base_data)} documents')
    return base_data

def init_document_searcher_en_wiki(searcher_args, args):
    # init the document searcher
    base_data = load_base_data('en_wiki')
    searcher = Searcher(
        'IVF10000,PQ16',
        dimension=768,
        nprobe=100
    )
    pretrained_model_name = searcher_args['pretrained_model'].replace('/', '_')
    model_name = searcher_args['model']
    if args['partial'] > 0:
        searcher.load(
            f'{args["root_dir"]}/data/en_wiki/{model_name}_{pretrained_model_name}_faiss_{args["partial"]}_percent.ckpt',
            f'{args["root_dir"]}/data/en_wiki/{model_name}_{pretrained_model_name}_corpus_{args["partial"]}_percent.ckpt'        
        )
    print(f'[!] init en-wiki searcher over; index partial: {args["partial"]}')
    return searcher, base_data

def init_document_searcher_bm25(searcher_args, dataset, args):
    # init the document searcher
    base_data = load_base_data(dataset)
    searcher = ESSearcher(f'{dataset}_phrase-copy', q_q=True)
    print(f'[!] load the BM25 searcher for {dataset}')
    return searcher, base_data


def init_document_searcher(searcher_args, dataset, args):
    # init the document searcher
    base_data = load_base_data(dataset)
    searcher = Searcher(
        searcher_args['index_type'],
        dimension=searcher_args['dimension'],
        nprobe=searcher_args['index_nprobe']
    )
    pretrained_model_name = searcher_args['pretrained_model'].replace('/', '_')
    model_name = searcher_args['model']
    searcher.load(
        f'{args["root_dir"]}/data/{dataset}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{dataset}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    )
    # searcher_args['local_rank'] = 0
    # searcher_agent = load_model(searcher_args)
    # save_path = f'{args["root_dir"]}/ckpt/{searcher_args["dataset"]}/{searcher_args["model"]}/best_{pretrained_model_name}_{searcher_args["version"]}.pt'
    # searcher_agent.load_model(save_path)
    # print(f'[!] init the searcher and dual-bert model over')
    return searcher, base_data

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

    # random.seed(args['seed'])
    # torch.manual_seed(args['seed'])
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    print(f'[!] init the copygeneration over')

    # build the search agent
    if args['dataset'] in ['en_wiki']:
        en_wiki_searcher, en_wiki_base_data = init_document_searcher_en_wiki(searcher_args, args)
        agent.model.init_searcher_en_wiki(en_wiki_searcher, en_wiki_base_data)
    else:
        # searcher, base_data = init_document_searcher(searcher_args, args['dataset'], args)
        searcher, base_data = init_document_searcher_bm25(searcher_args, args['dataset'], args)
        agent.model.init_searcher(searcher, base_data)

    searcher_args['local_rank'] = 0
    searcher_agent = load_model(searcher_args)
    pretrained_model_name = searcher_args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{searcher_args["dataset"]}/simcse/best_{pretrained_model_name}_{searcher_args["version"]}.pt'
    searcher_agent.load_model(save_path)
    print(f'[!] init the searcher and dual-bert model over')
    agent.model.init_searcher_agent(searcher_agent)

    # end2end faiss searcher
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
            batch['document'] = data['document']
            batch['prefix'] = data['prefix']
            batch['ground_truth'] = data['ground_truth']
            batch['temp'] = data['temp']
            batch['beam_search_size'] = data['beam_search_size']
            print(f'[!] beam search size: {batch["beam_search_size"]}')

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
            batch['use_phrase_cache'] = data['use_phrase_cache']
            batch['head_weight'] = data['head_weight']
            batch['tail_weight'] = data['tail_weight']
            batch['coarse_score_alpha'] = data['coarse_score_alpha']
            batch['coarse_score_softmax_temp'] = data['coarse_score_softmax_temp']
            
            if 'recall_topk' in data:
                agent.model.args['recall_topk'] = data['recall_topk']
            res, phrase_rate, time_cost = agent.model.work(batch) 
            succ = True
        except Exception as error:
            print(error)
            succ = False
            res = None
        result = {
            'result': res,
            'prefix': data['prefix'],
            'ground-truth': data['ground_truth'],
            'phrase-rate': phrase_rate,
            'succ': succ,
            'time-cost': time_cost,
        }
        return jsonify(result)

    return app

if __name__ == "__main__":
    args = vars(parser_args())
    app = create_app(**args)
    app.run(host='0.0.0.0', port=args['port'])
