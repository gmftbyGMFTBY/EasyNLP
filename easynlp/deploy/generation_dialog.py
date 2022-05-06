from header import *
from model import *
from config import *
from dataloader import *
from .utils import *


class DeployGenerationDialogAgent:

    def __init__(self, args):
        self.agent = load_model(args) 
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
        self.agent.load_model(save_path)
        print(f'[!] load model from {save_path}')
        self.args = args

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @timethis
    def work(self, batches):
        # contrastive search diversity
        gs = self.agent.generate_dialog(batches)
        return gs
