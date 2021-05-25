from header import *

'''init the search engine and collect the bm25 hard negative samples for the context in the corpus'''

def load_dataset(path):
    with open(path) as f:
        responses, dataset = [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            utterances = [''.join(uttr.split()) for uttr in line[1:]]
            if line[0] == '1':
                ctx = ' '.join(utterances[:-1])
                dataset.append((ctx, utterances[-1]))
            responses.append(utterances[-1])
    responses = list(set(responses))
    print(f'[!] read {len(responses)} responses from {path}')
    print(f'[!] read {len(dataset)} context from {path}')
    return dataset, responses


class ESUtils:
    
    def __init__(self, index_name, create_index=False):
        # self.es = Elasticsearch(http_auth=('elastic', 'elastic123'))
        self.es = Elasticsearch()
        self.index = index_name
        if create_index:
            if index_name in ['douban', 'ecommerce']:
                mapping = {
                    'properties': {
                        'utterance': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_smart'
                        },
                    }
                }
            else:
                mapping = {
                    'properties': {
                        'utterance': {
                            'type': 'text',
                        },
                    }
                }
                
            if self.es.indices.exists(index=self.index):
                print(f'[!] delete the index of the elasticsearch')
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert_utterances(self, utterances):
        # count = self.es.count(index=self.index)['count']
        actions = []
        for i, utterance in enumerate(tqdm(utterances)):
            actions.append({
                '_index': self.index,
                # '_id': i + count,
                'utterance': utterance,
            })
        helpers.bulk(self.es, actions) 
        time.sleep(5)
        print(f'[!] database size: {self.es.count(index=self.index)["count"]}')

class ESChat:
    
    def __init__(self, index_name):
        # self.es = Elasticsearch(http_auth=('elastic', 'elastic123'))
        self.es = Elasticsearch()
        self.index = index_name
        self.es.indices.put_settings(
            index=self.index,
            body={
                'index': {
                    'max_result_window': 500000,
                }
            }
        )
        
    def multi_search(self, querys, samples=10):
        # limit the querys length
        querys = [i[-150:] for i in querys]
        search_arr = []
        for query in querys:
            search_arr.append({'index': self.index})
            search_arr.append({'query': {'match': {'utterance': query}}, 'size': samples})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='douban')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--inner_bsz', type=int, default=128)
    args = vars(parser.parse_args())

    path = f'data/{args["dataset"]}/train.txt'
    dataset, responses = load_dataset(path)

    esutils = ESUtils(args["dataset"], create_index=True)

    # init the corpus
    esutils.insert_utterances(responses)

    # search
    eschat = ESChat(args['dataset'])
    collector = []
    for idx in tqdm(list(range(0, len(dataset), args['inner_bsz']))):
        ctx, res = [], []
        for c, r in dataset[idx:idx+args['inner_bsz']]:
            ctx.append(c)
            res.append(r)
        rest = eschat.multi_search(ctx, samples=args['topk']+1)
        rest = rest['responses']
        for sample, groundtruth in zip(rest, res):
            utterances = [i['_source']['utterance'] for i in sample['hits']['hits']]
            if res in utterances:
                utterances.remove(res)
            collector.append(utterances[:args['topk']])

    torch.save(collector, f'data/{args["dataset"]}/candidates.pt')
