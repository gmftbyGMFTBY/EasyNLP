from header import *
from .inference_knnlm_dataloader import *
from .post_dialog_pretrain_dataloader import *
from .inference_dialog_pretrain_dataloader import *
from .generative_dialog_pretrain_dataloader import *
from .retrieval_dialog_pretrain_dataloader import *
from .acc_test_dataloader import *
from .magic_contrastive_search_dataloader import *
from .inference_copygeneration_dataloader import *
from .filter_scorer_dataloader import *
from .filter_scorer_inference_dataloader import *
from .inference_copygeneration_dataloader import *
from .doctttttquery_dataloader import *
from .copygeneration_dataloader import *
from .colbert_dataloader import *
from .traditional_response_selection_dataloader import *
from .gpt2_dialog_dataloader import *
from .gpt2_contrastive_search_dataloader import *
from .gpt2_contrastive_search_super_large_dataloader import *
from .writer_rank_dataloader import *
from .mutual_dataloader import *
from .dual_bert_hier_dataloader import *
from .dual_bert_session_dataloader import *
from .dual_bert_curriculum_learning_dataloader import *
from .horse_human_test_dataloader import *
from .time_evaluation_dataloader import *
from .fine_grained_test_dataloader import *
from .bert_mask_augmentation_dataloader import *
from .gpt2_dataloader import *
from .gpt2_memory_dataloader import *
from .simcse_dataloader import *
from .post_train_dataloader import *
from .dual_bert_dataloader import *
from .hash_bert_dataloader import *
from .dual_bert_unsup_dataloader import *
from .gpt2_tacl_dataloader import *
from .dual_bert_pt_dataloader import *
from .dual_bert_full_dataloader import *
from .dual_bert_arxiv_dataloader import *
from .sa_bert_dataloader import *
from .bert_ft_dataloader import *
from .bart_ft_dataloader import *
from .bert_ft_scm_dataloader import *
from .bert_ft_auxiliary_dataloader import *
from .bert_ft_compare_dataloader import *
from .inference_dataloader import *
from .inference_full_filter_dataloader import *
from .inference_phrase_dataloader import *
from .inference_full_dataloader import *
from .inference_ctx_dataloader import *
from .dual_bert_full_wr_dataloader import *
from .inference_full_wr_dataloader import *

def load_dataset(args):
    if args['mode'] in ['train', 'test', 'valid']:
        dataset_name = args['models'][args['model']]['dataset_name']
        dataset_t = globals()[dataset_name]
    elif args['mode'] in ['inference']:
        # inference
        dataset_name = args['models'][args['model']]['inference_dataset_name']
        dataset_t = globals()[dataset_name]
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')

    if args['mode'] in ['inference']:
        path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    else:
        path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.txt'
    try:
        if args['model'] in ['bart-ft', 'dialog-eva']:
            vocab = BertTokenizer.from_pretraiend(args['tokenizer'])
        else:
            vocab = AutoTokenizer.from_pretrained(args['tokenizer'])
    except:
        vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])

    data = dataset_t(vocab, path, **args)

    if args['mode'] in ['train', 'inference']:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    try:
        if not os.path.exists(data.pp_path):
            data.save()
    except Exception as e:
        pass
    return data, iter_, sampler
