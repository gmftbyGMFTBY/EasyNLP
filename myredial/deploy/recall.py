from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher


def init_recall(args):
    searcher = Searcher(args['index_type'], dimension=args['dimension'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/context_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/context_corpus.ckpt',
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

    def work(self, batch):
        '''batch: a list of string (query)'''
        batch = [i['str'] for i in batch]
        vectors = self.agent.encode_queries(batch)    # [B, E]
        rest = self.searcher._search(vectors, topk=self.args['topk'])
        return rest
