from header import *

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecommerce')
    parser.add_argument('--dim', default=768, type=int)
    return parser.parse_args()

class Searcher:

    def __init__(self, dimension=768, nlist=100):
        self.searcher = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension), dimension, nlist, int(dimension/8), 8
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
            jobib.dump(self.corpus, f)

    def load(self, path_faiss, path_corpus):
        self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)

if __name__ == "__main__":
    args = vars(parser_args())

