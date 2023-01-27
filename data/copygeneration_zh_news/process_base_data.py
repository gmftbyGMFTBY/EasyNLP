from tqdm import tqdm

with open('train.txt') as f:
    dataset = [line.strip() for line in f.readlines()] 

with open('base_data.txt', 'w') as f:
    counter = 0
    for item in tqdm(dataset):
        string = f'{item}\t{counter},0\n'
        f.write(string)
        counter += 1
