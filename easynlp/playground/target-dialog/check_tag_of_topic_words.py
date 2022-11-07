import nltk
import ipdb

with open('hard_target.txt') as f:
    tags = []
    for line in f.readlines():
        _, topic_word = line.strip().split('\t')
        tags.extend([tag for _, tag in nltk.pos_tag(nltk.word_tokenize(topic_word))])

    tags = list(set(tags))
    print(tags)

