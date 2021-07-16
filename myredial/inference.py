from header import *
from dataloader import *
from model import *
from config import *
from inference_strategy import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--nums', type=int)
    parser.add_argument('--cut_size', type=int, default=500000)
    # inference context parameters
    parser.add_argument('--work_mode', type=str, default='response')    # response or context
    parser.add_argument('--pool_size', type=int, default=200)
    return parser.parse_args()


class Searcher:

    '''If q-q is true, the corpus is a list of tuple(context, response);
    If q-r is true, the corpus is a list of strings;
    
    Source corpus is a dict:
        key is the title, value is the url(maybe the name)
    if with_source is true, then self.if_q_q is False (only do q-r matching)'''

    def __init__(self, index_type, dimension=768, q_q=False, with_source=False):
        if index_type.startswith('BHash') or index_type in ['BFlat']:
            binary = True
        else:
            binary = False
        if binary:
            self.searcher = faiss.index_binary_factory(dimension, index_type)
        else:
            self.searcher = faiss.index_factory(dimension, index_type)
        self.corpus = []
        self.binary = binary
        self.with_source = with_source
        self.source_corpus = {}
        self.if_q_q = q_q

    def _build(self, matrix, corpus, source_corpus=None):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        if self.with_source:
            self.source_corpus = source_corpus
        print(f'[!] build collection with {self.searcher.ntotal} samples')

    def _search(self, vector, topk=20):
        D, I = self.searcher.search(vector, topk)
        if self.with_source:
            # pack up the source information and return
            # return the tuple (text, title, url)
            rest = [[(self.corpus[i][0], self.corpus[i][1], self.source_corpus[self.corpus[i][1]]) for i in N] for N in I]
        elif self.if_q_q:
            # the response is the second item in the tuple
            rest = [[self.corpus[i][1] for i in N] for N in I]
        else:
            rest = [[self.corpus[i] for i in N] for N in I]
        return rest

    def save(self, path_faiss, path_corpus, path_source_corpus=None):
        if self.binary:
            faiss.write_index_binary(self.searcher, path_faiss)
        else:
            faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)
        if self.with_source:
            with open(path_source_corpus, 'wb') as f:
                joblib.dump(self.source_corpus, f)

    def load(self, path_faiss, path_corpus, path_source_corpus=None):
        if self.binary:
            self.searcher = faiss.read_index_binary(path_faiss)
        else:
            self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)
        if self.with_source:
            with open(path_source_corpus, 'rb') as f:
                self.source_corpus = joblib.load(f)

    def add(self, vectors, texts):
        '''the whole source information are added in _build'''
        self.searcher.add(vectors)
        self.corpus.extend(texts)
        print(f'[!] add {len(texts)} dataset over')

    def move_to_gpu(self, device=0):
        # self.searcher = faiss.index_cpu_to_all_gpus(self.searcher)
        res = faiss.StandardGpuResources()
        self.searcher = faiss.index_cpu_to_gpu(res, device, self.searcher)
        print(f'[!] move index to GPU device: {device} over')


def inference(**args):
    work_mode = args['work_mode']
    data, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')

    if work_mode in ['writer-inference']:
        # load the pre-trained model on writer dataset
        agent.load_model(f'{args["root_dir"]}/ckpt/writer/{args["model"]}/best_{pretrained_model_name}.pt')
    else:
        agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    if work_mode in ['response']:
        agent.inference(data_iter, size=args['cut_size'])
    elif work_mode in ['writer-inference']:
        agent.inference_writer(data_iter, size=args['cut_size'])
    elif work_mode in ['context', 'gray-one2many', 'gray']:
        # gray and gray-one2many will use the checkpoint generated by the context work_mode
        agent.inference_context(data_iter)


if __name__ == "__main__":
    args = vars(parser_args())
    bert_fp_args = deepcopy(args)
    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    inference(**args)

    # barries
    torch.distributed.barrier()

    if args['work_mode'] not in ['gray-one2many'] and args['local_rank'] != 0:
        # gray-one2many may need the multiple gpus to speed up
        exit()

    # only the main process will run the following inference strategies
    if args['work_mode'] in ['writer-inference']:
        writer_with_source_strategy(args)
    elif args['work_mode'] in ['response']:
        response_strategy(args)
    elif args['work_mode'] in ['gray']:
        gray_strategy(args)
    elif args['work_mode'] in ['gray-one2many']:
        # 1. run response
        # 2. run gray-one2many will also tun the context work mode
        gray_one2many_strategy(args)
    elif args['work_mode'] == 'context':
        context_strategy(args)
    else:
        raise Exception(f'[!] Unknown work mode: {args["work_mode"]}')

