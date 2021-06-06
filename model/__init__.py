from .InteractionModels import *
from .RepresentationModels import *
from .LatentInteractionModels import *

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
    elif args['model'] in ['sa-bert-neg', 'sa-bert']:
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
    elif args['model'] == 'dual-bert-mlm':
        model = BERTDualMLMEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-semi':
        model = BERTDualSemiEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-writer':
        model = BERTDualWriterEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-ma':
        model = BERTDualMAEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'hash-bert':
        model = HashBertAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-pretrain':
        model = BERTDualPretrainEncoderAgent(
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
    elif args['model'] == 'dual-bert-kw':
        model = BERTDualKWEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-cross':
        model = BERTDualCrossEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-scm':
        model = BERTDualCompEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-fg':
        model = BERTDualFGEncoderAgent(
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
    elif args['model'] == 'dual-gru-hierarchical-trs':
        model = GRUDualHierarchicalTrsEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-hierarchical-trs':
        model = BERTDualHierarchicalTrsEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path']
        )
    elif args['model'] == 'dual-bert-hierarchical-trs-kd':
        teacher_model_path = f'ckpt/{args["dataset"]}/dual-bert/best.pt'
        model = BERTDualHierarchicalTrsKDEncoderAgent(
            args['multi_gpu'], 
            args['total_step'], 
            args['warmup_step'], 
            run_mode=args['mode'], 
            pretrained_model=args['pretrained_model'],
            local_rank=args['local_rank'], 
            dataset_name=args['dataset'],
            pretrained_model_path=args['pretrained_model_path'],
            teacher_model_path=teacher_model_path,
        )
    elif args['model'] == 'dual-bert-cl':
        model = BERTDualCLEncoderAgent(
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
