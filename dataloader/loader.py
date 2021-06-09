from header import *
from .dual_bert_dataloader import *
from .sa_bert_dataloader import *
from .bert_ft_dataloader import *

def load_dataset(args):
    MAP_ITEM = {
        'sa-bert': SABERTWithNegDataset,
        'bert-ft': BERTFTDataset,
        'dual-bert': BERTDualDataset,
        'poly-encoder': BERTDualDataset,
        'dual-bert-gray': BERTDualWithNegDataset,
        'dual-bert-hierarchical-trs': BERTDualHierarchicalDataset,
    }
    MAP = {
        'train': deepcopy(MAP_ITEM),
        'test': deepcopy(MAP_ITEM),
        'inference': deepcopy(MAP_ITEM),
    } 
    MAP['inference']['dual-bert'] = BERTDualInferenceContextResponseDataset

    path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.txt'
    vocab = BertTokenizer.from_pretrained(args['pretrained_model'])
        
    data = MAP[args['mode']][args['model']](vocab, path, **args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        data,
        num_replicas=dist.get_world_size(),
        rank=args['local_rank'],
    )
    iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
    if args['mode'] == 'train':
        sampler = train_sampler
    else:
        sampler = None

    if not os.path.exists(data.pp_path):
        data.save()
    return data, iter_, sampler
