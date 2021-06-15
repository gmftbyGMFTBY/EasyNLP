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


def read_text_data_with_neg(path, lang='zh'):
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


def read_text_data_dual_bert(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = ' [SEP] '.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_response_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')[-1]
            if lang == 'zh':
                utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset


def read_response_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = [line.strip() for line in f.readlines()]
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset
