from header import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecommerce')
    parser.add_argument('--nums', default=4, type=int)
    return parser.parse_args()


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
    args = vars(parser_args())
    queries, answers, order = [], [], []
    queries_text, answers_text = [], []
    for i in tqdm(range(args['nums'])):
        query, answer, q_text, a_text, q_order = torch.load(f'{args["root_dir"]}/data/{args["dataset"]}/inference_{i}.pt')
        answers.append(answer)
        queries.append(query)
        order.extend(q_order)
        queries_text.extend(q_text)
        answers_text.extend(a_text)
    order = np.argsort(order)
    answers = np.concatenate(answers)
    queries = np.concatenate(queries)
    queries = torch.from_numpy(np.array([queries[i] for i in order]))
    answers = torch.from_numpy(np.array([answers[i] for i in order]))
    queries_text = [queries_text[i] for i in order]
    answers_text = [answers_text[i] for i in order]
    
    searcher = Searcher()
    assert len(queries) == len(answers)
    assert len(queries_text) == len(answers_text)
    searcher._build(queries.cpu().numpy(), [(c, r) for c, r in zip(queries_text, answers_text)])
    
    searcher.save(
        f'data/{args["dataset"]}/context_faiss.ckpt',
        f'data/{args["dataset"]}/context_corpus.ckpt',
    )
    print(f'[!] save faiss index over')

    # search for each context
    print(f'[!] begin to search the candidates')
    candidates = []
    for idx in tqdm(range(0, len(queries), args['inner_bsz'])):
        q_matrix = queries[idx:idx+args['inner_bsz']].cpu().numpy()
        q_text_mapping_rest = queries_text[idx:idx+args['inner_bsz']]
        rests = searcher._search(q_matrix, topk=args['topk']+1)
        for gt, rest in zip(q_text_mapping_rest, rests):
            if gt in rest:
                rest.remove(gt)
            candidates.append(rest[:args['topk']])
    torch.save(candidates, f'data/{args["dataset"]}/candidates.pt')
    print(f'[!] save retrieved candidates into data/{args["dataset"]}/candidates.pt')
