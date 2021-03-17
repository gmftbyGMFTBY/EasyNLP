from header import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecommerce')
    parser.add_argument('--nums', default=4, type=int)
    parser.add_argument('--inner_bsz', default=128, type=int)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--pre_extract', default=50, type=int)
    return parser.parse_args()


class MemoryBank:

    def __init__(self, corpus):
        self.data = {}
        for idx, _ in tqdm(corpus.items()):
            self.data[idx] = torch.randn(768).half()
        print(f'[!] init the Memory Bank over')

    def update(self, ids, embds):
        embds = embds.cpu().detach()
        for idx, embd in zip(ids, embds):
            self.data[idx] = embd

    def search(self, ids, topk, bsz):
        # return B*[K, 768]
        keys = list(set(self.data) - set(ids))
        rest = []
        for _ in range(bsz):
            index = random.sample(keys, topk)
            embds = torch.stack([self.data[i] for i in index])    # [K, 768]
            rest.append(embds)
        rest = torch.stack(rest).permute(0, 2, 1)    # [B, 768, K]
        if torch.cuda.is_available():
            rest = rest.cuda()
        return rest


class Searcher:

    def __init__(self, dimension=768, nlist=100):
        # ========== IVFPQ ========== #
        quantizer = faiss.IndexFlatIP(dimension)
        self.searcher = faiss.IndexIVFPQ(
           quantizer, dimension, 100, int(dimension/8), 8
        )
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
    # reconstruct
    args = vars(parser_args())
    queries, q_text, query_order, matrixes, corpus, q_text_mapping = [], [], [], [], [], []
    for i in tqdm(range(args['nums'])):
        query, q_text_, q_text_r, q_order, matrix, text = torch.load(
            f'data/{args["dataset"]}/inference_{i}.pt'
        )
        q_text.extend(q_text_)
        matrixes.append(matrix)
        queries.append(query)
        query_order.extend(q_order)
        corpus.extend(text)
        q_text_mapping.extend(q_text_r)
    query_order = np.argsort(query_order)
    matrix = np.concatenate(matrixes)
    queries = np.concatenate(queries)
    queries = np.array([queries[i] for i in query_order])
    q_text = [q_text[i] for i in query_order]
    q_text_mapping = [q_text_mapping[i] for i in query_order]
        
    searcher = Searcher()
    assert len(matrix) == len(corpus)
    searcher._build(matrix, corpus)
    searcher.save(
        f'data/{args["dataset"]}/faiss.ckpt',
        f'data/{args["dataset"]}/corpus.ckpt',
    )
    print(f'[!] load checkpoint from {args["nums"]} files, and save them into data/{args["dataset"]}/faiss.ckpt and data/{args["dataset"]}/corpus.ckpt')

    # ========== Search ========== #
    print(f'[!] begin to search the candidates')
    candidates = []
    assert args['pre_extract'] > args['topk'], f'pre extracted samples must bigger than topk'
    for idx in tqdm(range(0, len(queries), args['inner_bsz'])):
        q_matrix = queries[idx:idx+args['inner_bsz']]
        q_text_mapping_rest = q_text_mapping[idx:idx+args['inner_bsz']]
        # rest = searcher._search(q_matrix, topk=args['topk']+1)
        rest = searcher._search(q_matrix, topk=args['pre_extract'])
        # reconstruct
        rr = []
        for item_gt, item_rest in zip(q_text_mapping_rest, rest):
            # ipdb.set_trace()
            if item_gt in item_rest:
                item_rest.remove(item_gt)
            # rr.append(item_rest[-args['topk']:])
            rr.append(item_rest[:args['topk']])
            # rr.append(random.sample(item_rest, args['topk']))
        candidates.extend(rr)
    torch.save(candidates, f'data/{args["dataset"]}/candidates.pt')
    print(f'[!] save retrieved candidates into data/{args["dataset"]}/candidates.pt')
