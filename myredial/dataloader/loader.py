from header import *
from .dual_bert_dataloader import *
from .simcse_dataloader import *
from .simcse_inference_dataloader import *
from .dual_bert_full_dataloader import *
from .sa_bert_dataloader import *
from .bert_ft_dataloader import *
from .inference_dataloader import *
from .inference_full_dataloader import *
from .inference_ctx_dataloader import *

def load_dataset(args):
    if args['mode'] in ['train', 'test']:
        dataset_name = args['models'][args['model']]['dataset_name']
        dataset_t = globals()[dataset_name]
    elif args['mode'] in ['inference']:
        # inference
        dataset_name = args['models'][args['model']]['inference_dataset_name']
        dataset_t = globals()[dataset_name]
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')

    path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.txt'

    if args['xlm']:
        vocab = XLMRobertaTokenizerFast.from_pretrained(args['tokenizer'])
    else:
        vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        
    data = dataset_t(vocab, path, **args)
    if args['mode'] in ['train', 'inference']:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data,
            num_replicas=dist.get_world_size(),
            rank=args['local_rank'],
        )
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    try:
        if not os.path.exists(data.pp_path):
            data.save()
    except:
        pass
    return data, iter_, sampler
