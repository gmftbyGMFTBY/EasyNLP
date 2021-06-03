from header import *
from dataloader import *
from model import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--neg_bsz', type=int, default=64)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--res_max_len', type=int, default=256)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='zh')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--head_num', type=int, default=5)
    parser.add_argument('--extra_tm', type=int, default=16)
    parser.add_argument('--total_steps', type=int, default=100000)
    return parser.parse_args()


def obtain_steps_parameters(train_data, args):
    args['total_step'] = len(train_data) * args['epoch'] // args['batch_size'] // (args['multi_gpu'].count(',') + 1)
    # args['total_step'] = args['total_steps']
    args['warmup_step'] = int(0.1 * args['total_step'])


def main(**args):
    if args['mode'] in ['train', 'train-post', 'train-dual-post']:
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        train_data, train_iter, sampler = load_dataset(args)
        
        # load test dataset
        test_args = deepcopy(args)
        test_args['mode'] = 'test'
        test_args['batch_size'] = 1
        test_data, test_iter, _ = load_dataset(test_args)

        obtain_steps_parameters(train_data, args)
        agent = load_model(args)
        if args['mode'] == 'train-dual-post':
            agent.load_model(f'ckpt/{args["dataset"]}/dual-bert/best.pt')
        agent.test_iter = test_iter
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        for epoch_i in range(args['epoch']):
            sampler.set_epoch(epoch_i)    # shuffle for DDP
            train_loss = agent.train_model(
                train_iter, 
                mode='train',
                recoder=sum_writer,
                idx_=epoch_i,
            )
            # only one process save the checkpoint
            if args['local_rank'] == 0:
                agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
            # test
            (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model()
            sum_writer.add_scalar(f'test-epoch/R10@1', r10_1, epoch_i)
            sum_writer.add_scalar(f'test-epoch/R10@2', r10_2, epoch_i)
            sum_writer.add_scalar(f'test-epoch/R10@5', r10_5, epoch_i)
            sum_writer.add_scalar(f'test-epoch/MRR', mrr, epoch_i)
            sum_writer.add_scalar(f'test-epoch/P@1', p1, epoch_i)
            sum_writer.add_scalar(f'test-epoch/MAP', MAP, epoch_i)
        sum_writer.close()
    elif args['mode'] == 'test':
        test_data, test_iter, _ = load_dataset(args)
        args['total_step'], args['warmup_step'] = 0, 0
        agent = load_model(args)
        agent.test_iter = test_iter
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model()
        print(f'R10@1: {round(r10_1, 4)}; R10@2: {round(r10_2, 4)}; R10@5: {round(r10_5, 4)}; MRR: {mrr}; P@1: {p1}; MAP: {MAP}')
    elif args['mode'] == 'inference':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        # inference the dataset and generate the vector for each sample
        _, (iter_res, iter_ctx), _ = load_dataset(args)
        # _, data_iter, _ = load_dataset(args)
        args['total_step'], args['warmup_step'] = 0, 0
        agent = load_model(args)
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        agent.inference(iter_res, iter_ctx)
        # agent.inference_qa(data_iter)
    elif args['mode'] == 'inference_qa':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        _, data_iter, _ = load_dataset(args)
        args['total_step'], args['warmup_step'] = 0, 0
        agent = load_model(args)
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        agent.inference_qa(data_iter)

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
