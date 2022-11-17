# evaluate the smoothness with the state-of-the-art cross-encoder model BERT-FP
import pprint
from header import *
from config import *
from model import *
from dataloader import *

def generate_mask(ids, pad_token_idx=0):
    '''generate the mask matrix of the ids, default padding token idx is 0'''
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] = 0.
    return mask

def make_batch(tokenizer, examples, eos_token_id, cls_token_id, sep_token_id, pad_token_id, max_len):

    ids, tids = [], []
    for example in examples:
        a, b = make_one_instance(tokenizer, example, eos_token_id, cls_token_id, sep_token_id, max_len)
        ids.append(a)
        tids.append(b)
    ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
    tids = pad_sequence(tids, batch_first=True, padding_value=pad_token_id)
    mask = generate_mask(ids, pad_token_idx=pad_token_id)
    return {'ids': ids.cuda(), 'tids': tids.cuda(), 'mask': mask.cuda()}


def make_one_instance(tokenizer, text_lists, eos_token_id, cls_token_id, sep_token_id, max_len):
    text_ids = [tokenizer.encode(text, add_special_tokens=False) for text in text_lists]
    context = []
    for ids in text_ids[:-1]:
        context.extend(ids + [eos_token_id])
    context.pop()
    response = text_ids[-1]
    truncate_pair(context, response, max_len)
    ids = [cls_token_id] + context + [sep_token_id] + response + [sep_token_id]
    tids = [0] * (len(context) + 2) + [1] * (len(response) + 1)
    ids = torch.LongTensor(ids)
    tids = torch.LongTensor(tids)
    return ids, tids

def main(path):
    with open(path) as f:
        sessions = [session.strip() for session in f.read().split('\n\n') if session.strip()]
        datasets = []
        for session in sessions:
            utterances = session.split('\n')
            context = [utterances[0].replace('[Context]', '').strip()]
            for utterance in utterances[2:]:
                utterance = re.sub('\[.*\]', '', utterance).strip()
                if utterance.startswith('chatbot'):
                    utterance = utterance.replace('chatbot:', '').strip()
                    # save a example
                    context.append(utterance)
                    datasets.append(deepcopy(context))
                else:
                    utterance = utterance.replace('human:', '').strip()
                    context.append(utterance)
        print(f'[!] collect {len(datasets)} evaluation samples')

    # evaluation
    batch_size = 128
    results = []
    for idx in tqdm(range(0, len(datasets), batch_size)):
        examples = datasets[idx:idx+batch_size]
        batch = make_batch(agent.vocab, examples, agent.eos, agent.cls, agent.sep, agent.pad, 256)
        with torch.no_grad():
            scores = F.softmax(agent.model(batch), dim=-1)[:, 1].cpu().tolist()
        results.extend(scores)
    return np.mean(results)

# prepare the model
test_args = {
    'mode': 'test',
    'dataset': 'tgconv',
    'model': 'bert-ft',
    'version': 300
}
config = load_config(test_args)
test_args.update(config)

agent = load_model(test_args)
pretrained_model_name = test_args['pretrained_model'].replace('/', '_')
save_path = f'{test_args["root_dir"]}/ckpt/{test_args["dataset"]}/{test_args["model"]}/best_{pretrained_model_name}_{test_args["version"]}.pt'
print('load model from', save_path)
agent.load_model(save_path)

results = {'TGConv': {}, 'TGCP': {}}
# for task in ['TGConv', 'TGCP']:
for task in ['TGConv']:
    if task == 'TGConv':
        for baseline in ['dkrn', 'kernel', 'matrix', 'neural', 'retrieval', 'retrieval_stgy', 'topkg']:
            # for mode in ['hard', 'easy']:
            for mode in ['hard']:
                path = f'playground/target-dialog/TGDR/{task}/{baseline}/{mode}.txt'
                try:
                    result = main(path)
                except:
                    print(f'[!] cannot find file: {path}')
                    continue
                print(f'[!] {path}: {round(result, 4)}')
                if baseline not in results[task]:
                    results[task][baseline] = {'hard': None, 'easy': None}
                results[task][baseline][mode] = result
    else:
        # for baseline in ['ours', 'dkrn', 'kernel', 'matrix', 'neural', 'retrieval', 'retrieval_stgy', 'topkg']:
        for baseline in ['topkg']:
            path = f'playground/target-dialog/TGDR/{task}/{baseline}/result.txt'
            try:
                result = main(path)
            except:
                print(f'[!] cannot find file: {path}')
                continue
            print(f'[!] {path}: {round(result, 4)}')
            if baseline not in results[task]:
                results[task][baseline] = {'result': None}
            results[task][baseline]['result'] = result

pprint.pprint(results)
    

