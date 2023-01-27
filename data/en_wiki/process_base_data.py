from tqdm import tqdm
import spacy

if __name__ == "__main__":
    with open('train.txt') as f:
        dataset = [line.strip() for line in f.readlines()]

    with open('base_data.txt', 'w') as f:
        counter = 0
        for line in tqdm(dataset):
            string = f'{line.strip()}\t{counter},0\n'
            f.write(string)
            counter += 1
            


