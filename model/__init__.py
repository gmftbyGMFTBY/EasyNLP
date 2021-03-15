from .bert_ft import *
from .sa_bert import *
from .bert_gen import *
from .bert_gen_ft import *
from .dual_bert import *
from .dual_bert_jsd import *
from .dual_bert_gen import *
from .dual_bert_adv import *
from .dual_bert_mb import *
from .dual_bert_one2many import *
from .dual_bert_hierarchical import *
from .dual_bert_poly import *
from .dual_bert_cl import *
from .dual_bert_vae import *
from .dual_bert_vae2 import *
from .searcher import *

def load_model(args):
    if args['model'] in ['bert-ft-multi', 'bert-ft']:
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
    elif args['model'] == 'sa-bert':
        model = SABERTFTAgent(
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
    elif args['model'] == 'dual-bert-adv':
        model = BERTDualAdvEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-gen':
        model = BERTDualGenEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-jsd':
        model = BERTDualJSDEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
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
    elif args['model'] == 'dual-bert-mb':
        model = BERTDualMBEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-one2many':
        model = BERTDualOne2ManyEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path'],
            head=args['head_num']
        )
    elif args['model'] == 'dual-bert-hierarchical':
        model = BERTDualHierarchicalEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-cl':
        model = BERTDualEncoderCLAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-vae':
        model = BERTDualEncoderVAEAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-vae2':
        model = BERTDualEncoderVAE2Agent(
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
