from header import *
from model import *
from config import *
from dataloader import *
from .utils import *


class DeployGenerationAgent:

    def __init__(self, args):
        self.agent = load_model(args) 
        # pretrained_model_name = args['pretrained_model'].replace('/', '_')
        # save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
        # self.agent.load_model(save_path)
        self.args = args

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @timethis
    def work(self, batches):
        seed = 1000 * random.uniform(0, 1)
        # contrastive search diversity
        self.set_seed(seed)
        g1 = self.agent.generate(batches)
        # topk topp sampling
        self.set_seed(seed)
        batches['decoding_method'] = 'topk_topp_repetition_penalty_batch_fast_search'
        g3 = self.agent.generate(batches)
        # contrastive search reference
        batches['decoding_method'] = 'contrastive_batch_search'
        batches['generation_num'] = 1
        batches['sampling_prefix_len'] = -1.
        g2 = self.agent.generate(batches)

        del batches['generation_num']
        return g1, g2, g3
