from header import *
from dataloader import *
from model import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--session_num', type=int, default=10)
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_len', type=int, default=256)
    return parser.parse_args()


def load_start_data():
    pass


def main(**args):
    args['total_step'], args['warmup_step'] = 0, 0
    agent = load_model(args)
    agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')

    # data
    
    # another agent
    agent = load_model(args)
    agent_.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')

    # simulat self-talk
    count_t = 0
    for _ in range(args['session_num']):
        pass


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
