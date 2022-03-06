from tqdm import tqdm
import json

def remove_empty_line(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            if line.strip():
                dataset.append(line)
    return dataset

def rewrite(data, path):
    with open(path, 'w') as f:
        for item in tqdm(data):
            item = json.loads(item)
            item = {'content': ''.join(item["q"])}
            item = json.dumps(item)
            f.write(f"{item}\n")


if __name__ == "__main__":
    # data = remove_empty_line('train_.txt')
    # rewrite(data, 'train.txt')

    data = remove_empty_line('test_.txt')
    rewrite(data, 'test.txt')
