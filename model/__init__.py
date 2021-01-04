from .bert_ft import *

def load_model(args):
    if args['model'] == 'bert-ft':
        model = BERTFTAgent(args['multi_gpu'], run_mode=args['mode'], lang=args['lang'], local_rank=args['local_rank'])
    return model