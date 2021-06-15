from header import *
from dataloader import *
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--nums', type=int)
    return parser.parse_args()


class Searcher:

    def __init__(self, index_type, dimension=768):
        self.searcher = faiss.index_factory(dimension, index_type)
        self.corpus = []

    def _build(self, matrix, corpus):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        print(f'[!] build collection with {self.searcher.ntotal} samples')

    def _search(self, vector, topk=20):
        queries = len(vector)
        _, I = self.searcher.search(vector, topk)
        rest = [[self.corpus[i] for i in N] for N in I]
        return rest

    def save(self, path_faiss, path_corpus):
        faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)

    def load(self, path_faiss, path_corpus):
        self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)


def inference(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('inference', args)

    _, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    agent.inference(data_iter)


if __name__ == "__main__":
    args = vars(parser_args())
    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('inference', args) 
    
    inference(**args)

    # barries
    torch.distributed.barrier()

    if args['local_rank'] == 0:
        embds, texts = [], []
        for i in tqdm(range(args['nums'])):
            embd, text = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_{i}.pt'
            )
            embds.append(embd)
            texts.extend(text)
        embds = np.concatenate(embds)
        
        searcher = Searcher(args['index_type'], dimension=args['dimension'])
        searcher._build(embds, texts)
        searcher.save(
            f'{args["root_dir"]}/data/{args["dataset"]}/context_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/context_corpus.ckpt',
        )
        print(f'[!] save faiss index over')
