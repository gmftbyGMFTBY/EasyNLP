from header import *
from dataloader import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()

def main(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    
    args['total_step'] = 10000
    args['warmup_step'] = 10000
    agent = load_model(args)
    onnx_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/onnx_ckpt.onnx'
    agent.model.convert_to_onnx(onnx_path)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
