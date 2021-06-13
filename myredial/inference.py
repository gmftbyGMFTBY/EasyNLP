from header import *
from dataloader import *
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()


def main(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('inference', args)

    _, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    agent.inference(data_iter)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
