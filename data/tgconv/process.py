import json

with open('concepts_nv.json') as f:
    dataset = []
    for line in f.readlines():
        item = json.loads(line)
        target = item['easy_target']
        context = item['dialog'][:1]

        dataset.append((target, context))

with open('easy_target.txt', 'w') as f:
    for item in dataset:
        string = '\t'.join(item[1]) + '\t' + item[0] + '\n'
        f.write(string)
