from header import *
from copy import deepcopy
from dataloader import *
from model import *
from config import *
from inference import *
from es import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    return parser.parse_args()

def select_topic_and_memory(k2m):
    # generating topics can be found in playground/target-dialog
    keys = list(k2m.keys())
    k = random.choice(keys)
    m = k2m[k]
    return k, m

def main_target_dialog(**args):
    args['mode'] = 'test'
    args['model'] = 'target-dialog'
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    # load dataset
    with open(f'{args["root_dir"]}/data/{args["dataset"]}/test.txt') as f:
        lines = f.readlines()
        test_iter = []
        for i in range(0, len(lines), 10):
            line = lines[i].strip().split('\t')[1:-1]
            test_iter.append(line)
        print(f'[!] load {len(test_iter)} sessions')

    # load the model 
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    print(f'[!] init the agent over')

    f = open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_target_dialog.txt', 'w')
    k2m = torch.load(f'{args["root_dir"]}/data/{args["dataset"]}/k2m.pt')
    success_num = 0
    for context_list in tqdm(test_iter):
        # random select the topic and memory
        topic, memory = select_topic_and_memory(k2m)
        if len(memory) > agent.args['max_memory_size']:
            memory = random.sample(memory, agent.args['max_memory_size'])

        agent.init(memory, topic, context_list[:-1])
        # utterance = context_list[-1]
        dialog, dialog_history = [], deepcopy(context_list)
        for turn_id in tqdm(range(agent.args["max_turn_num"])):
            # candidate, dis_1 = agent.work([utterance])
            candidate, dis_1 = agent.work(dialog_history)
            dialog_history.append(candidate)
            if topic in candidate:
                dialog.append(('chatbot', candidate, dis_1))
                success_num += 1
                break
            # utterance, dis_2 = agent.work_no_topic([candidate])
            utterance, dis_2 = agent.work_no_topic(dialog_history)
            dialog_history.append(utterance)
            if topic in utterance:
                dialog.append(('human', utterance, dis_2))
                success_num += 1
                break

            dialog.append(('chatbot', candidate, dis_1))
            dialog.append(('human', utterance, dis_2))

        # write the log
        context = ' [SEP] '.join(context_list)
        f.write(f'[Context] {context}\n')
        f.write(f'[Topic] {topic}\n')
        counter = 0
        for label, u, dis in dialog:
            f.write(f'[Distance {round(dis, 2)}] {label}: {u}\n')
            counter += 1
        f.write('\n')
        f.flush()
    f.close()
    print(f'[!] success rate: {round(success_num/len(test_iter), 2)}')

if __name__ == "__main__":
    args = vars(parser_args())
    main_target_dialog(**args)
