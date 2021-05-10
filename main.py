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
    return parser.parse_args()


def obtain_steps_parameters(train_data, args):
    args['total_step'] = len(train_data) * args['epoch'] // args['batch_size'] // (args['multi_gpu'].count(',') + 1)
    args['warmup_step'] = int(0.1 * args['total_step'])


def main(**args):
    if args['mode'] in ['train', 'train-post', 'train-dual-post']:
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        train_data, train_iter, sampler = load_dataset(args)
        train_bsz, mode = args['batch_size'], args['mode']
        
        # load test dataset
        args['mode'] = 'test'
        args['batch_size'] = 1
        test_data, test_iter, _ = load_dataset(args)

        args['mode'] = mode 
        args['batch_size'] = train_bsz

        obtain_steps_parameters(train_data, args)
        agent = load_model(args)
        agent.test_iter = test_iter
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        for i in tqdm(range(args['epoch'])):
            sampler.set_epoch(i)    # shuffle for DDP
            train_loss = agent.train_model(
                train_iter, 
                mode='train',
                recoder=sum_writer,
                idx_=i,
            )
            # only one process save the checkpoint
            if args['local_rank'] == 0:
                agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
            # test
            rest_path = f'rest/{args["dataset"]}/{args["model"]}/rest_epoch_{i}.txt'
            (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model()
            sum_writer.add_scalar(f'test-epoch/R10@1', r10_1, i)
            sum_writer.add_scalar(f'test-epoch/R10@2', r10_2, i)
            sum_writer.add_scalar(f'test-epoch/R10@5', r10_5, i)
            sum_writer.add_scalar(f'test-epoch/MRR', mrr, i)
            sum_writer.add_scalar(f'test-epoch/P@1', p1, i)
            sum_writer.add_scalar(f'test-epoch/MAP', MAP, i)
        sum_writer.close()
    elif args['mode'] == 'test':
        test_data, test_iter, _ = load_dataset(args)
        args['total_step'], args['warmup_step'] = 0, 0
        agent = load_model(args)
        agent.test_iter = test_iter
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        rest_path = f'rest/{args["dataset"]}/{args["model"]}/rest.txt'
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
