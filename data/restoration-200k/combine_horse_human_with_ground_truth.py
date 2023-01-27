import pickle
import ipdb

rrs_plus = pickle.load(open('horse-human-test.pkl', 'rb'))

with open('ft_local/new_test.txt') as f:
    dataset = {}
    for line in f.readlines():
        items = line.strip().split('\t')
        ctx, res = items[:-1], items[-1]
        dataset['\t'.join(ctx)] = res
    print(f'[!] load {len(dataset)} from test.txt')

# find ground-truth
new_rrs_plus = []
for session in rrs_plus:
    string = '\t'.join(session['ctx'])
    if string in dataset:
        new_rrs_plus.append({
            'res': session['res'],
            'ctx': session['ctx'],
            'gt': dataset[string]
        })
    else:
        pass

with open('new-horse-human-test.pkl', 'wb') as f:
    pickle.dump(new_rrs_plus, f)
print(f'[!] load {len(new_rrs_plus)} samples')
