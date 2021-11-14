import pickle
import ipdb
from tqdm import tqdm

def load_test(path, split=10):
    with open(path) as f:
        data = [line.strip().split('\t') for line in f.readlines()]
        dataset = {}
        for i in range(0, len(data), split):
            dataset[' [SEP] '.join(data[i][1:-1])] = data[i:i+split]
    print(f'[!] collect {len(dataset)} samples')
    assert len(dataset) == 1000
    return dataset

if __name__ == "__main__":
    data = load_test('test.txt', split=10)
    select_index = pickle.load(open('select_test_set_ctx.pkl', 'rb'))
    dataset = []
    for ctx in tqdm(select_index):
        if '我有c乳你要不' in ctx:
            dataset.append(data['o'+' [SEP] '.join(ctx)])
        else:
            dataset.append(data[' [SEP] '.join(ctx)])
    # write
    with open('test_select.txt', 'w') as f:
        for item in dataset:
            for line in item:
                string = '\t'.join(line) + '\n'
                f.write(string)
