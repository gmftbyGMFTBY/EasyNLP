import pickle
import ipdb
import json

with open('new-horse-human-test.pkl', 'rb') as f:
    data = pickle.load(f)
    
with open('test.txt', 'w') as f:
    for line in data:
        context = '|||'.join(line['ctx'])
        for res in line['res']:
            item = {
                'context': context,
                'ground_truth': line['gt'],
                'response': res[0],
                'score': res[1]
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')
