from tqdm import tqdm
import torch

def load_file(path):
    with open(path) as f:
        dataset = []    # essay
        for line in f.readlines():
            line = line.strip().split('\t')
            essay_id, passage_id, _, sentence_id, _, sentence = line
            essay_id = int(essay_id)
            passage_id = int(passage_id)
            sentence_id = int(sentence_id)
            sentence = sentence.replace('|', '')
            if len(sentence) < 5:
                continue
            dataset.append((
                essay_id, passage_id, sentence_id, sentence    
            ))
        print(f'[!] collect {len(dataset)} sentences')

    train_dataset = []
    essay_point, passage_point = 0, 0
    for i in range(1, len(dataset)):
        if dataset[i-1][0] == dataset[i][0]:
            if dataset[i-1][1] == dataset[i][1]:
                # same essay, same passage, add positive sample
                train_dataset.append((1, dataset[i-1][-1], dataset[i][-1]))
                for j in range(passage_point, i-1):
                    train_dataset.append((0, dataset[j][-1], dataset[i][-1]))
            else:
                passage_point = i
        else:
            essay_point = i
    print(f'[!] collect {len(train_dataset)} training samples')
    return train_dataset

def write_file(dataset, path):
    with open(path, 'w') as f:
        for label, s1, s2 in dataset:
            f.write(f'{label}\t{s1}\t{s2}\n')

if __name__ == "__main__":
    data = load_file('full.txt')
    write_file(data, 'train.txt')
