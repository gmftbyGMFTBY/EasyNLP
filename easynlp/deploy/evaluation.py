from header import *
from model import *
from config import *
from dataloader import *
from .utils import *


class DeployEvaluationAgent:

    def __init__(self, args):
        self.agent = load_model(args) 
        self.args = args
        # load the parameters
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
        self.agent.load_model(save_path)
        print(f'[!] load parameters from {save_path}')

    @timethis
    def work(self, batches):
        item_list = []
        for batch in batches:
            context, candidate1, candidate2 = batch['context'], batch['candidate1'], batch['candidate2']
            score = self.agent.compare_candidates(context, candidate1, candidate2)
            batch['score'] = round(score, 4)
            item_list.append(batch)
        return item_list
