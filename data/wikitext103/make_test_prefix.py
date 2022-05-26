from tqdm import tqdm
import json
import spacy

with open('test.txt') as f:
    dataset = []
    for line in f.readlines():
        line = line.strip()
        if len(line.split()) > 32:
            dataset.append(line)
    print(f'[!] collect {len(dataset)} valid samples')

nlp = spacy.load('en_core_web_sm')
with open('test_prefix.txt', 'w') as f:
    prefix_length_ratio = 0.5
    for line in tqdm(dataset):
        doc = nlp(line)
        tokens = [token.text for token in doc]
        prefix_length = int(len(tokens) * prefix_length_ratio)
        prefix = ' '.join(tokens[:prefix_length])
        gt = ' '.join(tokens[prefix_length:])
        item = {'prefix': prefix, 'ground_truth': gt}
        item = json.dumps(item, ensure_ascii=False)
        f.write(f'{item}\n')

