from header import *
from dataloader import *
from model import *
from config import *
from cluster_utils import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--nums', type=int)
    parser.add_argument('--cluster_batch_size', type=int)
    parser.add_argument('--ncluster', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parser_args())
    cluster_on_context_and_response_space(args)
