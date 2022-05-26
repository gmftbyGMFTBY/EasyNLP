from tqdm import tqdm

def load_dataset(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip()
            if line and line.startswith('=') is False:
                dataset.append(line)
    print(f'[!] collect {len(dataset)} samples')
    return dataset

def write_dataset(path, data):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(line + '\n')

if __name__ == "__main__":
    # train
    dataset = load_dataset('wiki.train.tokens')
    write_dataset('train.txt', dataset)


    dataset = load_dataset('wiki.valid.tokens')
    write_dataset('valid.txt', dataset)

    
    dataset = load_dataset('wiki.test.tokens')
    write_dataset('test.txt', dataset)

    
