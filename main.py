from header import *
from dataloader import *
from model import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    return parser.parse_args()

def main(**args):
    if args['mode'] == 'train':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        train_data, train_iter = load_dataset(args)
        
        # total step
        args['total_step'] = len(train_data) * args['epoch'] // args['batch_size']
        args['warmup_step'] = int(0.1 * args['total_step'] / (args['multi_gpu'].count(',') + 1))
        agent = load_model(args)
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        for i in tqdm(range(args['epoch'])):
            train_loss = agent.train_model(
                train_iter, 
                mode='train',
                recoder=sum_writer,
                idx_=i,
            )
            # only one process save the checkpoint
            if args['local_rank'] == 0:
                agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        sum_writer.close()
    else:
        test_data, test_iter = load_dataset(args)
        args['total_step'], args['warmup_step'] = 0, 0
        agent = load_model(args)
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        rest_path = f'rest/{args["dataset"]}/{args["model"]}/rest.txt'
        test_loss = agent.test_model(test_iter, rest_path)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    print('[!] parameters:')
    print(args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    main(**args)
