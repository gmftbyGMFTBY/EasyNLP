from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='writer')
    parser.add_argument('--model', type=str, default='dual-bert-gray')
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parser_args())

    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('search', args)

    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/context_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/context_corpus.ckpt',
    )
    print(f'[!] load faiss over')

    args['mode'] = 'search'
    agent = load_model(args) 
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    print(f'[!] load model over')

    # test
    dataset = read_json_data_dual_bert(
        f'{args["root_dir"]}/data/{args["dataset"]}/train.txt',
        lang='zh',
    )
    for i in range(0, len(dataset), 32):
        texts = [sample[1] for sample in dataset[i:i+32]]
        responses = [sample[2] for sample in dataset[i:i+32]]
        ipdb.set_trace()
        vectors = agent.encode_queries(texts) 
        rests = searcher._search(vectors, topk=args['topk'])

