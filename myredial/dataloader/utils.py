from header import *


def read_json_data_dual_bert(path, lang='zh'):
    with open(path) as f:
        dataset = []
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            context = ' [SEP] '.join(item['q'])
            response = item['r'].strip()
            dataset.append((1, context, response))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            context = item['q']
            response = item['r'].strip()
            # NOTE: the candidates may not be 10 (less than 10)
            candidates = [i.strip() for i in item['nr'] if i.strip()]
            dataset.append((context, response, candidates))
            responses.extend([response] + candidates)
        responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} unique utterances in {path}')
    return dataset, responses


def read_text_data_utterances_compare_test(path, lang='zh', size=10):
    with open(path) as f:
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip().split('\t')[1:]
            responses.extend(line)
        responses_ = list(set(responses))

    with open(path) as f:
        dataset = []
        cache = []
        for line in tqdm(list(f.readlines())):
            if len(cache) == size:
                lines = [line.strip().split('\t') for line in cache]
                lines = [(line[0], line[1:]) for line in lines]     # (label, utterance list)

                context = lines[0][1][1:-1]
                responses = [line[1][-1] for line in lines if line[0] == '1']
                if len(responses) != 0:
                    response = random.choice(responses)    # random select one as the ground-truth
                    hard_negative_samples = [line[1][-1] for line in lines if line[0] == '0']
                    easy_negative_samples = random.sample(responses_, len(hard_negative_samples))
                    dataset.append((
                        context, 
                        response, 
                        hard_negative_samples, 
                        easy_negative_samples
                    ))
                else:
                    # ignore this case
                    pass
                cache = []
            cache.append(line)
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_text_data_utterances_compare(path, lang='zh'):
    '''read from the train_gray dataset'''
    with open(path) as f:
        responses = []
        for line in tqdm(list(f.readlines())):
            item = json.loads(line.strip())
            responses.extend(item['q'] + [item['r']] + item['nr'])
        responses = list(set(responses))

    with open(path) as f:
        dataset = []
        for line in tqdm(list(f.readlines())):
            item = json.loads(line.strip())
            context = item['q']
            response = item['r']
            hard_negative_samples = item['nr']
            easy_negative_samples = random.sample(responses, len(hard_negative_samples))
            dataset.append((context, response, hard_negative_samples, easy_negative_samples))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_text_data_utterances(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            dataset.append((label, utterances))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_text_data_with_neg_inner_session_neg(path, lang='zh'):
    with open(path) as f:
        dataset, responses = [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if label == 0:
                continue
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context = utterances[:-1]
            response = utterances[-1]
            candidates = utterances[:-1]
            dataset.append((context, response, candidates))
            responses.extend(utterances)
    responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} utterances from {path}')
    return dataset, responses


def read_text_data_with_neg_q_r_neg(path, lang='zh'):
    path = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path) as f:
        dataset, responses = [], []
        for line in f.readlines():
            line = json.loads(line.strip())
            context = line['q']
            response = line['r']
            candidates = line['nr']
            dataset.append((context, response, candidates))
            responses.extend(context + [response] + candidates)
        responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} utterances from {path}')
    return dataset, responses

def read_text_data_dual_bert(path, lang='zh'):
    sep = ' [SEP] '
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = sep.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_response_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')
            if int(utterance[0]) == 0:
                continue
            utterance = utterance[-1]
            if lang == 'zh':
                utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset


def read_cl_response_data(path, lang='zh', max_context_turn=0):
    with open(path) as f:
        dataset = {}
        for line in f.readlines():
            utterances = line.strip().split('\t')
            if int(utterances[0]) == 0:
                continue
            utterances = utterances[1:]
            if lang == 'zh':
                utterances = [''.join(i.split()) for i in utterances]

            if max_context_turn == 0:
                context = ' [SEP] '.join(utterances[:-1])
            else:
                context = ' [SEP] '.join(utterances[-max_context_turn-1:-1])

            response = utterances[-1]
            if response in dataset:
                dataset[response].append(context)
            else:
                dataset[response] = [context]
        print(f'[!] load {len(dataset)} responses from {path}')
        return dataset
            

def read_response_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        pbar = tqdm(f.readlines())
        for line in pbar: 
            dataset.append(line.strip())
        print(f'[!] already collect {len(dataset)} utterances for inference')
    return dataset


def read_text_data_with_source(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            context = line[0]
            response = json.loads(line[1])
            dataset.append(([context], response[0], response[1], response[2]))
        print(f'[!] collect: {len(dataset)} utterances for inference')
    return dataset
