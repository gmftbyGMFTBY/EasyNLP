from header import *
from dataloader import *
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    return parser.parse_args()


def main(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    print('test', args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    # (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model(test_iter, print_output=True)
    outputs = agent.test_model(test_iter, print_output=True)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
