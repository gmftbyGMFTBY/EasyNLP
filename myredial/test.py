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
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best.pt')
    (r10_1, r10_2, r10_5), mrr, p1, MAP = agent.test_model(test_iter)
    pprint.pprint(f'R10@1: {round(r10_1, 4)}; R10@2: {round(r10_2, 4)}; R10@5: {round(r10_5, 4)}; MRR: {mrr}; P@1: {p1}; MAP: {MAP}')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
