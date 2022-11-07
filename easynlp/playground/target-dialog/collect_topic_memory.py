from tqdm import tqdm
import nltk
import torch
from collections import Counter
from jieba import analyse
import ipdb
import jieba
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='restoration-200k', type=str)
    parser.add_argument('--topk', default=10000, type=int)
    return parser.parse_args()

def collect_topics(path):
    with open(path) as f:
        lines = f.readlines()
        dataset = []
        for line in lines:
            items = line.strip().split('\t')[1:]
            dataset.extend(items)
        dataset = list(set(dataset))
        print(f'[!] collect {len(dataset)} utterances')
    k2m = {}
    kws = []
    for utterance in tqdm(dataset):
        # chinese
        # keywords = analyse.extract_tags(utterance)

        keywords = [word for word, tag in nltk.pos_tag(nltk.word_tokenize(utterance)) if tag in ['VB', 'NN', 'JJ', 'VBG', 'NNS', 'RB', 'WRB']]

        kws.extend(keywords)
        for keyword in keywords:
            if keyword in k2m:
                k2m[keyword].append(utterance)
            else:
                k2m[keyword] = [utterance]
    return kws, k2m

if __name__ == '__main__':
    args = vars(parse_args())
    # path = f'/apdcephfs/share_916081/johntianlan/MyReDial/data/convai2_tgcp/test.txt'
    # kws, k2m = collect_topics(path)
    path = f'/apdcephfs/share_916081/johntianlan/MyReDial/data/convai2_tgcp/train.txt'
    kws, k2m = collect_topics(path)
    # kws = kws + kws_
    # k2m.update(k2m_)
    
    kws = Counter(kws)
    # kws = [word for word, _ in kws.most_common(n=args['topk'])]
    print(f'[!] collect {len(kws)} keywords')

    k2m = {k:k2m[k] for k in kws}
    torch.save(k2m, 'k2m.pt')
