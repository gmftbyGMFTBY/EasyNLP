from tqdm import tqdm

'''convert the train.txt file into the inference format dataset:
document_string\tdoc_id,chunk_id'''

def load_train(path):
    with open(path) as f:
        dataset = [line.strip() for line in f.readlines()]
    print(f'[!] load {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    dataset = load_train('train.txt')
    with open('base_data.txt', 'w') as f:
        for idx, line in tqdm(list(enumerate(dataset))):
            id = f'{idx},1'
            string = f'{line}\t{id}\n'
            f.write(string)
