from header import *
from dataloader import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()


def obtain_steps_parameters(train_data, args):
    args['total_step'] = len(train_data) * args['epoch'] // args['batch_size'] // (args['multi_gpu'].count(',') + 1)
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])


def main(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    test_args = deepcopy(args)

    args['mode'] = 'train'
    train_config = load_config(args)
    args.update(train_config)
    print(args)
    train_data, train_iter, sampler = load_dataset(args)

    test_args['mode'] = 'test'
    test_config = load_config(test_args)
    test_args.update(test_config)
    test_data, test_iter, _ = load_dataset(test_args)
    
    # set seed
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    obtain_steps_parameters(train_data, args)
    agent = load_model(args)
    
    sum_writer = SummaryWriter(
        log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}'
    )
    for epoch_i in range(args['epoch']):
        sampler.set_epoch(epoch_i)    # shuffle for DDP
        train_loss = agent.train_model(
            train_iter, 
            test_iter,
            recoder=sum_writer,
            idx_=epoch_i,
        )
        if args['local_rank'] == 0:
            agent.save_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        # test after epoch
        (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model(test_iter, test_args)
        sum_writer.add_scalar(f'test-epoch/R10@1', r10_1, epoch_i)
        sum_writer.add_scalar(f'test-epoch/R10@2', r10_2, epoch_i)
        sum_writer.add_scalar(f'test-epoch/R10@5', r10_5, epoch_i)
        sum_writer.add_scalar(f'test-epoch/MRR', mrr, epoch_i)
        sum_writer.add_scalar(f'test-epoch/P@1', p1, epoch_i)
        sum_writer.add_scalar(f'test-epoch/MAP', MAP, epoch_i)
    sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
