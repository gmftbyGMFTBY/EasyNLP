from header import *
from dataloader import *
from model import *
from config import *
from inference_strategy import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--nums', type=int)
    parser.add_argument('--gen_dataset_num', type=int, default=500000)
    parser.add_argument('--gen_dataset_ctx_length', type=int, default=5)
    parser.add_argument('--gen_dataset_topk', type=int, default=5)
    parser.add_argument('--gray_topk', type=int, default=5)
    parser.add_argument('--cut_size', type=int, default=500000)
    # inference context parameters
    parser.add_argument('--work_mode', type=str, default='response')    # response or context
    parser.add_argument('--pool_size', type=int, default=200)
    return parser.parse_args()


def inference(**args):
    work_mode = args['work_mode']
    data, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')

    if work_mode in ['writer-inference']:
        # load the pre-trained model on writer dataset
        agent.load_model(f'{args["root_dir"]}/ckpt/writer/{args["model"]}/best_{pretrained_model_name}.pt')
    else:
        agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    if work_mode in ['response']:
        agent.inference(data_iter, size=args['cut_size'])
        pass
    elif work_mode in ['writer-inference']:
        agent.inference_writer(data_iter, size=args['cut_size'])
    # elif work_mode in ['context', 'gray-one2many', 'gray', 'unparallel']:
    elif work_mode in ['context']:
        # gray and gray-one2many will use the checkpoint generated by the context work_mode
        agent.inference_context(data_iter, size=args['cut_size'])
    else:
        pass

if __name__ == "__main__":
    args = vars(parser_args())
    bert_fp_args = deepcopy(args)
    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    inference(**args)

    # barries
    torch.distributed.barrier()

    if args['local_rank'] != 0:
        if args['work_mode'] in ['self-play']:
            pass
        else:
            exit()

    # only the main process will run the following inference strategies
    if args['work_mode'] in ['writer-inference']:
        writer_with_source_strategy(args)
    elif args['work_mode'] in ['response']:
        response_strategy(args)
    elif args['work_mode'] in ['gray']:
        gray_strategy(args)
    elif args['work_mode'] in ['gray-one2many']:
        # 1. run response
        # 2. run gray-one2many will also tun the context work mode
        gray_one2many_strategy(args)
    elif args['work_mode'] == 'context':
        pass
    elif args['work_mode'] == 'unparallel':
        # response_strategy(args)
        print(f'[!] build index for responses over')
        unparallel_strategy(args)
    # elif args['work_mode'] in ['self-play', 'gray-extend']:
    elif args['work_mode'] in ['self-play', 'gray-extend']:
        gray_extend_strategy(args)
        torch.distributed.barrier()
        combine_all_generate_samples(args)
    else:
        raise Exception(f'[!] Unknown work mode: {args["work_mode"]}')

