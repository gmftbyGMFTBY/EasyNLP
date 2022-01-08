import ipdb
import jieba

def distinct_sentence_level_char(sentence, n=1):
    items = list(sentence)
    base = []
    for i in range(0, len(items) - n + 1):
        base.append(tuple(items[i:i+n]))
    try:
        return len(set(base)) / len(base)
    except:
        return 0.

def distinct_sentence_level_word(sentence, n=1):
    tokens = list(jieba.cut(sentence))
    base = []
    for i in range(0, len(tokens) - n + 1):
        base.append(tuple(tokens[i:i+n]))
    try:
        return len(set(base)) / len(base)
    except:
        return 0.
