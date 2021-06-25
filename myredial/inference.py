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
    # inference context parameters
    parser.add_argument('--work_mode', type=str, default='response')    # response or context
    parser.add_argument('--pool_size', type=int, default=200)
    return parser.parse_args()


class Searcher:

    '''If q-q is true, the corpus is a list of tuple(context, response);
    If q-r is true, the corpus is a list of strings'''

    def __init__(self, index_type, dimension=768, q_q=False):
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
        self.if_q_q = q_q

    def _build(self, matrix, corpus):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        print(f'[!] build collection with {self.searcher.ntotal} samples')

    def _search(self, vector, topk=20):
        D, I = self.searcher.search(vector, topk)
        if self.if_q_q:
            # the response is the second item in the tuple
            rest = [[self.corpus[i][1] for i in N] for N in I]
        else:
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

    def move_to_all_gpus(self):
        self.searcher = faiss.index_cpu_to_all_gpus(self.searcher)
        print(f'[!] move index to GPUs over')


def inference(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    work_mode = args['work_mode']

    data, data_iter, _ = load_dataset(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt')
    if work_mode == 'response':
        agent.inference(data_iter, size=args['cut_size'])
    elif work_mode in ['context', 'gray']:
        agent.inference_context(data_iter)


if __name__ == "__main__":
    args = vars(parser_args())
    args['mode'] = 'inference'
    config = load_config(args)
    args.update(config)
    print('inference', args) 
    
    inference(**args)

    # barries
    torch.distributed.barrier()

    if args['local_rank'] != 0:
        exit()

    # only the main process
    if args['work_mode'] == 'response':
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
    elif args['work_mode'] == 'gray':
        # collect the gray negative dataset
        embds, contexts, responses = [], [], []
        for i in tqdm(range(args['nums'])):
            embd, context, response = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}.pt'        
            )
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        embds = np.concatenate(embds) 

        # read faiss index
        model_name = args['model']
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        searcher = Searcher(args['index_type'], dimension=args['dimension'])
        searcher.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
        )
        # speed up with gpu
        searcher.move_to_all_gpus()
        print(f'[!] read the faiss index over, begin to search from the index')

        # search
        # NOTE: Make sure the responses are saved in the faiss index
        collection = []
        for i in tqdm(range(0, len(embds), args['batch_size'])):
            batch = embds[i:i+args['batch_size']]    # [B, E]
            context = contexts[i:i+args['batch_size']]
            response = responses[i:i+args['batch_size']]
            result = searcher._search(batch, topk=args['pool_size'])
            for c, r, rest in zip(context, response, result):
                if r in rest:
                    rest.remove(r)
                nr = random.sample(rest, args['topk'])
                collection.append({'q': c, 'r': r, 'nr': nr})

        # write into new file
        path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'
        with open(path, 'w') as f:
            for item in tqdm(collection):
                string = json.dumps(item)
                f.write(f'{string}\n')
    elif args['work_mode'] == 'context':
        # inference the context and do the q-q matching
        embds, corpus = [], []
        for i in tqdm(range(args['nums'])):
            embd, context, response = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}.pt'        
            )
            embds.append(embd)
            corpus.extend([(c, r) for c, r in zip(context, response)])
        embds = np.concatenate(embds) 
        # write into faiss index
        model_name = args['model']
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        searcher = Searcher(args['index_type'], dimension=args['dimension'], q_q=True)
        searcher._build(embds, corpus)
        searcher.save(
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_corpus.ckpt',
        )
        print(f'[!] save the q-q matching faiss over')
    else:
        raise Exception(f'[!] Unknown work mode: {args["work_mode"]}')

