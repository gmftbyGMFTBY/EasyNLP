from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher
from .utils import *


def init_recall(args):
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    model_name = args['model']
    pretrained_model_name = args['pretrained_model']
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    print(f'[!] load faiss over')
    agent = load_model(args) 
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    print(f'[!] load model over')
    return searcher, agent


class RecallAgent:

    def __init__(self, args):
        self.searcher, self.agent = init_recall(args)
        self.args = args

    @timethis
    def work(self, batch, topk=None):
        '''batch: a list of string (query)'''
        batch = [i['str'] for i in batch]
        vectors = self.agent.encode_queries(batch)    # [B, E]
        topk = topk if topk else self.args['topk']
        rest = self.searcher._search(vectors, topk=topk)
        return rest
