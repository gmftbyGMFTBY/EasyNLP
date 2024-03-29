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
    # if args['model'] in ['bert-ft-compare', 'bert-ft-compare-token']:
    if args['model'] in ['bert-ft-compare-token']:
        # each context contains `gray_cand_num` random negative and `gray_cand_num` hard negative samples
        args['total_step'] = len(train_data) * args['epoch'] * args['gray_cand_num'] * 2 // args['inner_bsz'] // (args['multi_gpu'].count(',') + 1)
    else:
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

    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    if args['model'] in args['no_train_models']:
        raise Exception(f'[!] {args["model"]} is not allowed to be trained')
    # print('train', args)
    train_data, train_iter, sampler = load_dataset(args)

    if args['model'] not in args['no_test_models']:
        config = load_config(test_args)
        test_args.update(config)

        if args['valid_during_training']:
            # valid set for training
            test_args['mode'] = 'valid'
        test_data, test_iter, _ = load_dataset(test_args)
    else:
        test_iter = None
    
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
        
    if args['is_step_for_training']:
        args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])
        agent = load_model(args)
        pbar = tqdm(total=args['total_step'])
        gobal_total_step, current_step, over_train_flag = 0, 0, False
        # 1000000 is the virtual epoch, only the step are used
        sampler.set_epoch(0)    # shuffle for DDP
        for _ in range(100000000):
            for batch in train_iter:
                agent.train_model(
                    batch, 
                    recoder=sum_writer, 
                    current_step=current_step, 
                    pbar=pbar
                )

                # if args['local_rank'] == 0 and current_step in args['test_step'] and current_step > 0:
                #     agent.test_now(test_iter, sum_writer)
                if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
                    pretrained_model_name = args['pretrained_model'].replace('/', '_')
                    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}_{current_step}.pt'
                    agent.save_model(save_path)
                    
                    # valiadtion the ppl
                    # if args['model'] in ['gpt2-original']:
                    #     ppl = []
                    #     for batch in tqdm(test_iter):
                    #         ppl_ = agent.model.module.calculate_ppl(batch['ids'], batch['ids_mask'], batch['label'])
                    #         ppl.append(ppl_)
                    #     ppl = np.mean(ppl)
                    #     sum_writer.add_scalar(f'test/ppl', ppl, current_step)
                    # print(f'[!] ppl on test set: {round(ppl, 4)}')

                current_step += 1
                if current_step > args['total_step']:
                    over_train_flag = True
                    break
            if over_train_flag:
                break
        if args['model'] == 'doctttttquery':
            path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/doctttttquery/best.pt'
            agent.save_model(path)
            print(f'[!] save model into: {path}')
    else:
        obtain_steps_parameters(train_data, args)
        agent = load_model(args)
        batch_num = 0
        for epoch_i in range(args['epoch']):
            sampler.set_epoch(epoch_i)    # shuffle for DDP
            nb = agent.train_model(
                train_iter, 
                test_iter,
                recoder=sum_writer,
                idx_=epoch_i,
                whole_batch_num=batch_num,
            )
            batch_num += nb
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
