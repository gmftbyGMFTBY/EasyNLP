from tqdm import tqdm
import random
import ipdb
import json

'''spite the raw text into the train set and test set'''


def load_raw_text(path):
    with open(path) as f:
        dataset = [[]]
        head = True
        pbar = tqdm(f.readlines())
        counter = 0
        for line in pbar:
            if head:
                head = False
                continue
            if line.strip():
                dataset[-1].append(line)
            else:
                dataset.append([])
                head = True
            counter += 1
            if counter % 1000 == 0:
                pbar.set_description(f'[!] documents: {len(dataset)}')
    return dataset


def write_into_json_file(dataset, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            item = {'q': item}
            f.write(json.dumps(item)+'\n')


if __name__ == "__main__":
    dataset = load_raw_text('gpt2_1000w_lines.txt')

    test_size = 1000
    test_idx = set(random.sample(range(len(dataset)), 1000))
    test_set = [dataset[i] for i in test_idx]
    train_set = [dataset[i] for i in range(len(dataset)) if i not in set(test_idx)]

    write_into_json_file(test_set, 'test.txt')
    write_into_json_file(train_set, 'train.txt')
