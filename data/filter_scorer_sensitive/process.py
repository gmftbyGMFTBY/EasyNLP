from tqdm import tqdm
import random

def load_file(path):
    with open(path) as f:
        pos, neg = [], []
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            label = line[-1]
            context = line[:-1]
            try:
                label = int(label)
            except:
                continue
            if label == 1:
                neg.append(context)
            else:
                pos.append(context)
        print(f'[!] pos|neg: {len(pos)}|{len(neg)}')
    return neg, pos


if __name__ == "__main__":
    random.seed(0.)
    neg, pos = load_file('sensitive_data_20200331.txt')

    test_size = 1000
    neg_test_index = list(range(len(neg)))
    neg_test_index = random.sample(neg_test_index, test_size)

    pos_test_index  = list(range(len(pos)))
    pos_test_index = random.sample(pos_test_index, test_size)

    neg_test_index, pos_test_index = set(neg_test_index), set(pos_test_index)

    neg_test_set = [neg[i] for i in neg_test_index]
    pos_test_set = [pos[i] for i in pos_test_index]

    assert len(neg_test_set) == len(pos_test_set) == test_size
    test_set = list(neg_test_set + pos_test_set)
    test_set_label = [0] * test_size + [1] * test_size
    test_index = list(range(test_size * 2))
    random.shuffle(test_index)
    test_set = [test_set[i] for i in test_index]
    test_set_label = [test_set_label[i] for i in test_index]

    # build the train set 
    neg_train = [neg[i] for i in range(len(neg)) if i not in neg_test_index]
    pos_train = [pos[i] for i in range(len(pos)) if i not in pos_test_index]
    pos_train = random.sample(pos_train, len(neg_train))

    train_set = neg_train + pos_train
    train_set_label = [0] * len(neg_train) + [1] * len(pos_train)
    train_index = list(range(len(train_set)))
    random.shuffle(train_index)
    train_set = [train_set[i] for i in train_index]
    train_set_label = [train_set_label[i] for i in train_index]

    # write into the file
    with open('train.txt', 'w', encoding='utf-8') as f:
        for c, l in zip(train_set, train_set_label):
            utterance = '\t'.join(c)
            string = f'{l}\t{utterance}\n'
            f.write(string)

    # write intot the sensitive test set data into the fil
    with open('test.txt', 'w', encoding='utf-8') as f:
        for c, l in zip(test_set, test_set_label):
            utterance = '\t'.join(c)
            string = f'{l}\t{utterance}\n'
            f.write(string)
