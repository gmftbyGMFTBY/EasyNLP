from .bert_ft import *
from .bert_gen import *
from .bert_gen_ft import *
from .dual_bert import *
from .dual_bert_poly import *
from .searcher import *

def load_model(args):
    if args['model'] == 'bert-ft':
        model = BERTFTAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'], 
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'bert-gen':
        model = BERTGenAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset']
        )
    elif args['model'] == 'bert-gen-ft':
        model = BERTGenFTAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'], 
            local_rank=args['local_rank'], 
            dataset_name=args['dataset']
        )
    elif args['model'] == 'dual-bert':
        model = BERTDualEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-poly':
        model = BERTPolyEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    else:
        raise Exception(f'[!] Unknow model: {args["model"]}')
    return model
