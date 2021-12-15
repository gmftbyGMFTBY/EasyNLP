def distinct_sentence_level(sentence):
    unique_chars = set(sentence)
    try:
        return len(unique_chars) / len(sentence)
    except:
        # divide zero
        return 0.

def distinct_sentence_level_n_gram(sentence):
    import jieba
    tokens = list(jieba.cut(sentence))
    try:
        return len(set(tokens)) / len(tokens)
    except:
        return 0.
