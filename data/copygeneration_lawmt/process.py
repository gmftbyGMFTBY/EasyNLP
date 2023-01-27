from tqdm import tqdm
import spacy

if __name__ == "__main__":
    with open('train.en') as f:
        dataset = [line.strip() for line in f.read().split('\n') if line.strip()]
    print(f'[!] collect {len(dataset)} samples')

    new_dataset = []
    for line in tqdm(dataset):
        if len(line.split()) < 10:
            continue
        new_dataset.append(line)
    print(f'[!] collect {len(new_dataset)} documents')

    with open('train_lawmt.txt', 'w') as f:
        for line in new_dataset:
            f.write(line + '\n')


