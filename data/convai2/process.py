import random
from itertools import chain
import json
import ipdb
from tqdm import tqdm

with open('train_both_original.txt') as f:

    dataset = []
    cache = {'persona': [], 'context': []}

    for line in tqdm(f.readlines()):
        if line.startswith('1 your persona:'):
            if len(cache['context']) > 0:
                dataset.append(cache)
                cache = {'persona': [], 'context': []}
        line = line.strip()
        if 'persona' in line:
            persona_index = line.index('persona')
            cache['persona'].append(line[persona_index+len('persona: '):].strip())
        else:
            line = line.split('\t')[:2]
            line[0] = ' '.join(line[0].split()[1:])
            cache['context'].append(line)
    print(f'[!] find {len(dataset)} sessions')

    utterances = []
    for session in dataset:
        utterances.extend(session['persona'])
        utterances.extend(list(chain(*session['context'])))
    utterances = list(set(utterances))
    print(f'[!] find {len(utterances)} utterances')

with open('train.txt', 'w') as f:
    for session in dataset:
        persona = session['persona']
        context = session['context']
        negative = random.choice(utterances)

        persona_string = '\t'.join(persona)
        string = persona_string + '\t'.join(list(chain(*context)))
        f.write(f'1\t{string}\n')
        string = persona_string + '\t'.join(list(chain(*context[:-1]))) + '\t' + context[-1][0] + '\t' + negative
        f.write(f'0\t{string}\n')

