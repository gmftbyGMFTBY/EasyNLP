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
    parser.add_argument('--total_workers', type=int)
    return parser.parse_args()


def obtain_steps_parameters(train_data, args):
    args['total_step'] = len(train_data) * args['epoch'] // args['batch_size'] // (args['multi_gpu'].count(',') + 1)
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])


def main(**args):
    torch.cuda.empty_cache()
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['global_rank'] = dist.get_rank()

    # test set configuration
    test_args = deepcopy(args)
    test_args['mode'] = 'test'
    config = load_config(test_args)
    test_args.update(config)

    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    if args['model'] in args['no_train_models']:
        raise Exception(f'[!] {args["model"]} is not allowed to be trained')
    # print('train', args)
    train_data, train_iter, sampler = load_dataset(args)
    test_data, test_iter, _ = load_dataset(test_args)
    
    # set seed
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    if args['local_rank'] == 0:
        sum_writer = SummaryWriter(
            log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["version"]}',
            comment=pretrained_model_name,
        )
    else:
        sum_writer = None
        
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])
    agent = load_model(args)
    pbar = tqdm(total=args['total_step'])
    gobal_total_step, current_step, over_train_flag = 0, 0, False
    sampler.set_epoch(0)    # shuffle for DDP
    if agent.load_last_step:
        current_step = agent.load_last_step + 1
    for _ in range(100000000):
        for batch in train_iter:
            agent.train_model(
                batch, 
                recoder=sum_writer, 
                current_step=current_step, 
                pbar=pbar
            )
            if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
                # test the ppl
                ppls = []
                for test_batch in tqdm(test_iter):
                    ppl = agent.test_model_ppl_contrastive_search(test_batch)
                    ppls.append(ppl)
                ppl = np.mean(ppl)
                print(f'[!] PPL: {ppl}')
                sum_writer.add_scalar(f'test/ppl', ppl, current_step)
                # save the model
                pretrained_model_name = args['pretrained_model'].replace('/', '_')
                save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}_{current_step}.pt'
                agent.save_model_long(save_path, current_step)
            current_step += 1
            if current_step > args['total_step']:
                over_train_flag = True
                break
        if over_train_flag:
            break
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
