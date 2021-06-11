from header import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecommerce')
    parser.add_argument('--nums', default=4, type=int)
    parser.add_argument('--index_type', default='', type=str)
    parser.add_argument('--dimension', default=768, type=int)
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

if __name__ == "__main__":
    args = vars(parser_args())
    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('build_index': args)

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
