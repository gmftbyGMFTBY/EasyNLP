from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher
from .utils import *


class RerankAgent:

    def __init__(self, args):
        self.agent = load_model(args) 
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
        self.agent.load_model(save_path)
        self.args = args

    @timethis
    def work(self, batches):
        scores = self.agent.rerank(batches)
        return scores
