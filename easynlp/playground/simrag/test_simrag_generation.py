from header import *
from dataloader import *
from model import *
from config import *
from inference import *
from es import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def main_generation(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # test
    context_list = [
        '刺猬很可爱！以前别人送了只没养，味儿太大！', 
        '是很可爱但是非常臭', 
        '是啊，没办法养', 
        '那个怎么养哦不会扎手吗'
    ]

    retrieval_list = [
        '刺猬这种东西确实听扎手的，所以不太好养啊',
        '扎手啊'
    ]

    res = agent.simrag_talk(context_list, retrieval_list=retrieval_list)
    print(f'[Context  ] {context_list}')
    print(f'[Retrieval] {retrieval_list}')
    print(f'[Response ] {res}')


if __name__ == "__main__":
    args = vars(parser_args())
    main_generation(**args)
