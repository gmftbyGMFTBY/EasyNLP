from .bert_ft import *
from .bert_gen import *
from .bert_gen_ft import *

def load_model(args):
    if args['model'] == 'bert-ft':
        model = BERTFTAgent(args['multi_gpu'], args['total_step'], args['warmup_step'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'], dataset_name=args['dataset'])
    elif args['model'] == 'bert-gen':
        model = BERTGenAgent(args['multi_gpu'], args['total_step'], args['warmup_step'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'], dataset_name=args['dataset'])
    elif args['model'] == 'bert-gen-ft':
        model = BERTGenFTAgent(args['multi_gpu'], args['total_step'], args['warmup_step'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'], dataset_name=args['dataset'])
    return model