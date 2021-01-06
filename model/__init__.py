from .bert_ft import *
from .bert_gen import *

def load_model(args):
    if args['model'] == 'bert-ft':
        model = BERTFTAgent(args['multi_gpu'], args['total_step'], args['warmup_step'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'])
    elif args['model'] == 'bert-gen':
        model = BERTGenAgent(args['multi_gpu'], args['total_step'], args['warmup_step'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'])
    return model