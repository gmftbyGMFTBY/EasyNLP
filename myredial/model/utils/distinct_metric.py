def distinct_sentence_level(sentence):
    unique_chars = set(sentence)
    return len(unique_chars) / len(sentence)

def distinct_sentence_level_n_gram(sentence):
    import jieba
    tokens = list(jieba.cut(sentence))
    return len(set(tokens)) / len(tokens)
