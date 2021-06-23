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
    parser.add_argument('--cut_size', type=int, default=500000)
    return parser.parse_args()


class Searcher:

    def __init__(self, index_type, dimension=768):
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

    def _build(self, matrix, corpus):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        print(f'[!] build collection with {self.searcher.ntotal} samples')

    def _search(self, vector, topk=20):
        D, I = self.searcher.search(vector, topk)
        rest = [[self.corpus[i] for i in N] for N in I]
        return rest

    def save(self, path_faiss, path_corpus):
        if self.binary:
            faiss.write_index_binary(self.searcher, path_faiss)
        else:
            faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)

    def load(self, path_faiss, path_corpus):
        if self.binary:
            self.searcher = faiss.read_index_binary(path_faiss)
        else:
            self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)

    def add(self, vectors, texts):
        self.searcher.add(vectors)
        self.corpus.extend(texts)
        print(f'[!] add {len(texts)} dataset over')


def inference(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('inference', args)

    data, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    agent.inference(data_iter, size=args['cut_size'])

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
        already_added = []
        for i in tqdm(range(args['nums'])):
            for idx in range(100):
                try:
                    embd, text = torch.load(
                        f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                    )
                    print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt')
                except:
                    break
                embds.append(embd)
                texts.extend(text)
                already_added.append((i, idx))
            if len(embds) > 10000000:
                break
        embds = np.concatenate(embds) 
        searcher = Searcher(args['index_type'], dimension=args['dimension'])
        searcher._build(embds, texts)
        print(f'[!] train the searcher over')

        # add the external dataset
        for i in tqdm(range(args['nums'])):
            for idx in range(100):
                if (i, idx) in already_added:
                    continue
                try:
                    embd, text = torch.load(
                        f'{args["root_dir"]}/data/{args["dataset"]}/inference_{i}_{idx}.pt'
                    )
                    print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{i}_{idx}.pt')
                except:
                    break
                searcher.add(embd, text)
        print(f'[!] total samples: {searcher.searcher.ntotal}')

        model_name = args['model']
        pretrained_model_name = args['pretrained_model']
        searcher.save(
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
        )
        print(f'[!] save faiss index over')
