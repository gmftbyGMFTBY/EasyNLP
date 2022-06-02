from header import *
from dataloader import *
from model import *
from config import *
from inference import *
from es import *

'''
Test script:
    1. test the rerank performance [rerank]
    2. test the recall performance [recall]
    3. test the es recall performance [es_recall]
    4. test the comparison relationship among candidates [compare]

If you use the [recall], make sure the inference.sh has already been done
'''


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--recall_mode', type=str, default='q-r')
    parser.add_argument('--file_tags', type=str, default='22335,22336')
    parser.add_argument('--log', action='store_true', dest='log')
    parser.add_argument('--no-log', action='store_false', dest='log')
    parser.add_argument('--candidate_size', type=int, default=100)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def prepare_self_play_test_inference(**args):
    '''prepare the dataloader and the faiss index for recall test'''
    # ========== load the dataset ========== #
    args['mode'] = 'test'
    inf_args = deepcopy(args)
    partner_args = deepcopy(args)
    inf_partner_args = deepcopy(args)

    # use test mode args load test dataset and model
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # load the dataset
    test_data, test_iter, _ = load_dataset(args)

    # ========== load the partner agent ========== #
    partner_args['mode'] = 'test'
    partner_args['model'] = 'dual-bert'
    config = load_config(partner_args)
    partner_args.update(config)
    partner_agent = load_model(partner_args)
    pretrained_model_name = partner_args['pretrained_model'].replace('/', '_')
    save_path = f'{partner_args["root_dir"]}/ckpt/{partner_args["dataset"]}/{partner_args["model"]}/best_{pretrained_model_name}_200.pt'
    partner_agent.load_model(save_path)
    return test_iter, (agent, partner_agent)




def prepare_self_play_inference(**args):
    '''prepare the dataloader and the faiss index for recall test'''
    # ========== load the dataset ========== #
    args['mode'] = 'test'
    inf_args = deepcopy(args)
    partner_args = deepcopy(args)
    inf_partner_args = deepcopy(args)

    # use test mode args load test dataset and model
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # load the dataset
    test_data, test_iter, _ = load_dataset(args)

    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)
    model_name = inf_args['model']
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    faiss_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt'        
    corpus_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    searcher = Searcher(inf_args['index_type'], dimension=inf_args['dimension'], q_q=False)
    searcher.load(faiss_ckpt_path, corpus_ckpt_path)

    # ========== load the partner agent ========== #
    partner_args['mode'] = 'test'
    partner_args['model'] = 'dual-bert'
    config = load_config(partner_args)
    partner_args.update(config)
    partner_agent = load_model(partner_args)
    pretrained_model_name = partner_args['pretrained_model'].replace('/', '_')
    save_path = f'{partner_args["root_dir"]}/ckpt/{partner_args["dataset"]}/{partner_args["model"]}/best_{pretrained_model_name}_900.pt'
    partner_agent.load_model(save_path)

    inf_partner_args['mode'] = 'inference'
    config = load_config(inf_partner_args)
    inf_partner_args.update(config)
    model_name = inf_partner_args['model']
    pretrained_model_name = inf_partner_args['pretrained_model'].replace('/', '_')
    faiss_ckpt_path = f'{inf_partner_args["root_dir"]}/data/{inf_partner_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt'        
    corpus_ckpt_path = f'{inf_partner_args["root_dir"]}/data/{inf_partner_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    partner_searcher = Searcher(inf_partner_args['index_type'], dimension=inf_partner_args['dimension'], q_q=False)
    partner_searcher.load(faiss_ckpt_path, corpus_ckpt_path)
    return test_iter, (agent, partner_agent), (searcher, partner_searcher)


def prepare_inference(**args):
    '''prepare the dataloader and the faiss index for recall test'''
    # use test mode args load test dataset and model
    inf_args = deepcopy(args)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    # print('test', args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # ========== load the relevane evaluation model (dual-bert) ========== #
    inf_args_relevance = deepcopy(args)
    inf_args_relevance['mode'] = 'test'
    inf_args_relevance['model'] = 'dual-bert'
    config = load_config(inf_args_relevance)
    inf_args_relevance.update(config)
    relevance_agent = load_model(inf_args_relevance)
    pretrained_model_name = inf_args_relevance['pretrained_model'].replace('/', '_')
    save_path = f'{inf_args_relevance["root_dir"]}/ckpt/{inf_args_relevance["dataset"]}/{inf_args_relevance["model"]}/best_{pretrained_model_name}_{inf_args_relevance["version"]}.pt'
    relevance_agent.load_model(save_path)
    print(f'[!] build the relevance evaluation model over')

    # ========== load the ppl evaluation model (gpt2) ========== #
    inf_args_ppl = deepcopy(args)
    inf_args_ppl['mode'] = 'test'
    inf_args_ppl['model'] = 'gpt2-original'
    config = load_config(inf_args_ppl)
    inf_args_ppl.update(config)
    ppl_agent = load_model(inf_args_ppl)
    pretrained_model_name = inf_args_ppl['pretrained_model'].replace('/', '_')
    save_path = f'{inf_args_ppl["root_dir"]}/ckpt/{inf_args_ppl["dataset"]}/{inf_args_ppl["model"]}/best_{pretrained_model_name}_{inf_args_ppl["version"]}.pt'
    ppl_agent.load_model(save_path)
    print(f'[!] build the ppl evaluation model over')

    # ========== load the dataset ========== #
    test_data, test_iter, _ = load_dataset(args)

    # ===== use inference args ===== #
    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)
    # load faiss index
    model_name = inf_args['model']
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    # if inf_args['recall_mode'] == 'q-q':
    #     q_q = True
    #     faiss_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_faiss.ckpt'        
    #     corpus_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_corpus.ckpt'        
    # else:
    q_q = False
    faiss_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt'        
    corpus_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    searcher = Searcher(inf_args['index_type'], dimension=inf_args['dimension'], q_q=q_q)
    searcher.load(faiss_ckpt_path, corpus_ckpt_path)
    print(f'[!] load faiss over')
    return test_iter, inf_args, searcher, agent, relevance_agent, ppl_agent

def main_compare(**args):
    '''compare mode applications:
        1. replace with the human evaluation
        2. rerank agent for rerank agent'''
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    results = agent.compare_evaluation(test_iter)

    log_file_path = f'{args["root_dir"]}/data/{args["dataset"]}/test_compare_log.txt'
    avg_scores = []
    with open(log_file_path, 'w') as f:
        for item in results:
            f.write(f'[Context   ] {item["context"]}\n')
            f.write(f'[Response-1] {item["responses"][0]}\n')
            f.write(f'[Response-2] {item["responses"][1]}\n')
            f.write(f'[Comp-score] {item["score"]}\n\n')
            avg_scores.append(item['score'])
    avg_score = round(np.mean(avg_scores), 2)
    print(f'[!] Write the log into {log_file_path}')
    print(f'[!] Average Compare Scores: {avg_score}')


def main_rerank(**args):
    '''whether to use the rerank model'''
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)
    # print('test', args)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    if args['rank']:
        new_args['model'] = args['rank']
        config = load_config(new_args)
        new_args.update(config)
        print(f'[!] RERANK AGENT IS USED')
    else:
        new_args = None
        print(f'[!] RERANK AGENT IS NOT USED')

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # rerank model
    if new_args:
        rerank_agent = load_model(new_args)
        save_path = f'{new_args["root_dir"]}/ckpt/{new_args["dataset"]}/{new_args["model"]}/best_{pretrained_model_name}.pt'
        rerank_agent.load_model(save_path)
        print(f'[!] load rank order agent from: {save_path}')
    else:
        rerank_agent = None

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=False)
    cost_time = time.time() - bt
    cost_time *= 1000    # ms
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_rerank_{pretrained_model_name}_{args["version"]}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')
    print(f'Cost-Time: {round(cost_time, 2)} ms')


def main_es_recall(**args):
    args['model'] = 'dual-bert'
    test_iter, inf_args, _, agent, relevance_agent, ppl_agent = prepare_inference(**args)

    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    # inference
    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)
    inf_args['topk'] = inf_args['recall_topk']

    searcher = ESSearcher(
        f'{inf_args["dataset"]}_{inf_args["recall_mode"]}', 
        q_q=True
    )

    # test recall (Top-20, Top-100)
    pbar = tqdm(test_iter)
    counter, acc = 0, 0
    cost_time = []
    log_collector = []
    ppl, relevance = [], []
    for batch in pbar:
        if 'ids' in batch:
            context = agent.convert_to_text(batch['ids'])
        elif 'context' in batch:
            context = batch['context']
        else:
            raise Exception(f'[!] Error during test es recall')

        bt = time.time()
        rest = searcher.search(context, topk=inf_args['topk'])
        et = time.time()
        cost_time.append(et - bt)
        batch['candidates'] = rest

        gt_candidate = batch['text']
        if len(gt_candidate) == 0:
            continue
        for text in gt_candidate:
            if text in rest:
                acc += 1
                break
        counter += 1

        # 2. perplexity
        ppl_scores = ppl_agent.rerank(batch)
        ppl.append(np.mean(ppl_scores))

        # 3. relevance
        relevance_scores = relevance_agent.rerank_recall_evaluation(batch)
        relevance.append(np.mean(relevance_scores))

        pbar.set_description(f'[!] Top-{inf_args["topk"]}: {round(acc/counter, 4)}; Relevance-{inf_args["topk"]}: {round(np.mean(relevance)*100, 2)}; PPL-{inf_args["topk"]}: {round(np.mean(ppl), 4)}')
    
    topk_metric = round(acc/counter, 4)
    relevance_metric = round(np.mean(relevance)*100, 2)
    ppl_metric = round(np.mean(ppl), 4)
    avg_time = round(np.mean(cost_time)*1000, 2)    # ms
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    print(f'[!] Top-{inf_args["topk"]}: {topk_metric}')
    print(f'[!] Relevance-{inf_args["topk"]}: {relevance_metric}')
    print(f'[!] PPL-{inf_args["topk"]}: {ppl_metric}')
    print(f'[!] Average Times: {avg_time} ms')
    with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/test_result_recall_{pretrained_model_name}.txt', 'w') as f:
        print(f'Top-{inf_args["topk"]}: {topk_metric}', file=f)
        print(f'Average Times: {avg_time} ms', file=f)
    return 

def main_recall(**args):
    '''test the recall with the faiss index'''
    # use test mode args load test dataset and model
    test_iter, inf_args, searcher, agent, relevance_agent, ppl_agent = prepare_inference(**args)
    inf_args['topk'] = inf_args['recall_topk']

    # test recall (Top-20, Top-100)
    pbar = tqdm(test_iter)
    counter, acc = 0, 0
    ppl, relevance = [], []
    cost_time = []
    log_collector = []
    for batch in pbar:
        if args['model'] in ['dual-bert', 'hash-bert', 'lsh',  'bpr']:
            if 'ids' in batch:
                ids = batch['ids'].unsqueeze(0)
                ids_mask = torch.ones_like(ids)
            elif 'context' in batch:
                ids, ids_mask = agent.model.totensor([batch['context']], ctx=True)
            else:
                raise Exception(f'[!] process test dataset error')
            vector = agent.model.get_ctx(ids, ids_mask)   # [E]
        else:
            vector = agent.model.get_ctx(batch['ids'], batch['ids_mask'], batch['turn_length'])   # [E]

        # if not searcher.binary:
        try:
            vector = vector.cpu().numpy()
        except:
            pass

        bt = time.time()
        rest = searcher._search(vector, topk=inf_args['topk'])[0]
        et = time.time()
        cost_time.append(et - bt)
        batch['candidates'] = rest

        # evaluation the package
        # 1. top-k
        # if 'context' in batch:
        #     context = batch['context']
        # elif 'ids' in batch:
        #     context = agent.convert_to_text(batch['ids'])
        # log_collector.append({
        #     'context': context,
        #     'rest': rest
        # })

        gt_candidate = batch['text']
        if len(gt_candidate) == 0:
            continue
        for text in gt_candidate:
            if text in rest:
                acc += 1
                break
        counter += 1

        # 2. perplexity
        ppl_scores = ppl_agent.rerank(batch)
        ppl.append(np.mean(ppl_scores))

        # 3. relevance
        relevance_scores = relevance_agent.rerank_recall_evaluation(batch)
        relevance.append(np.mean(relevance_scores))

        pbar.set_description(f'[!] Top-{inf_args["topk"]}: {round(acc/counter, 4)}; Relevance-{inf_args["topk"]}: {round(np.mean(relevance)*100, 2)}; PPL-{inf_args["topk"]}: {round(np.mean(ppl), 4)}')

    topk_metric = round(acc/counter, 4)
    relevance_metric = round(np.mean(relevance)*100, 2)
    ppl_metric = round(np.mean(ppl), 4)
    avg_time = round(np.mean(cost_time)*1000, 2)    # ms
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    print(f'[!] Top-{inf_args["topk"]}: {topk_metric}')
    print(f'[!] Relevance-{inf_args["topk"]}: {relevance_metric}')
    print(f'[!] PPL-{inf_args["topk"]}: {ppl_metric}')
    print(f'[!] Average Times: {avg_time} ms')
    with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/test_result_recall_{pretrained_model_name}.txt', 'w') as f:
        print(f'Top-{inf_args["topk"]}: {topk_metric}', file=f)
        print(f'Average Times: {avg_time} ms', file=f)
    return 
    if args['log']:
        model_name = args['model']
        with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/recall_{model_name}_log.txt', 'w') as f:
            for item in log_collector:
                f.write(f'[Context] {item["context"]}\n')
                # DO NOT SAVE ALL THE RESULTS INTO THE LOG FILE (JUST TOP-10)
                for neg in item['rest'][:10]:
                    f.write(f'{neg}\n')
                f.write('\n')


def main_acc_test(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    agent.test_model_acc(test_iter)


def main_horse_human(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    collections = agent.test_model_horse_human(test_iter, print_output=True)
    ndcg_3, ndcg_5 = [], []
    for label, score in collections:
        group = [(l, s) for l, s in zip(label, score)]
        group = sorted(group, key=lambda x: x[1], reverse=True)
        group = [l for l, s in group]
        ndcg_3.append(NDCG(group, 3))
        ndcg_5.append(NDCG(group, 5))
    print(f'[!] NDCG@3: {round(np.mean(ndcg_3), 4)}')
    print(f'[!] NDCG@5: {round(np.mean(ndcg_5), 4)}')


def main_rerank_fg(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    collections = agent.test_model_fg(test_iter, print_output=True)
    for name in collections:
        sbm_scores, weighttau_scores = [], []
        for label, score in collections[name]:
            r1 = [(i, j) for i, j in zip(range(len(label)), label)]
            r1 = [i for i, j in sorted(r1, key=lambda x: x[1], reverse=True)]
            r2 = [(i, j) for i, j in zip(range(len(score)), score)]
            r2 = [i for i, j in sorted(r2, key=lambda x: x[1], reverse=True)]
            sbm_scores.append(SBM(r1, r2))
            weighttau_scores.append(kendalltau_score(r1, r2))
        print(f'[!] Set Based Metric of Annotator {name}: {round(np.mean(sbm_scores), 4)}')
        print(f'[!] Weighted Kendall Tau of Annotator {name}: {round(np.mean(weighttau_scores), 4)}')


def main_rerank_time(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=False, rerank_agent=None, core_time=True)
    cost_time = outputs['core_time']*1000
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_rerank_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')

def main_ppl(**args):
    '''test the ppl on the test dataset (GPT2, KNN-LM)'''
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    outputs = agent.test_model_ppl(test_iter, print_output=True)
    print(f'[!] PPL: {round(outputs, 4)}')

def main_generation(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=True)
    cost_time = time.time() - bt
    cost_time *= 1000    # ms
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_generation_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')
    print(f'Cost-Time: {round(cost_time, 2)} ms')


def main_self_play(**args):
    test_iter, (agent, partner_agent), (searcher, partner_searcher) = prepare_self_play_inference(**args)
    pbar = tqdm(test_iter)
    cost_time = []
    max_number = 0
    counter = 0
    agent_t1_avg, agent_t2_avg, partner_agent_t_avg = [], [], []
    max_turn_size = 70
    batch_size = 1
    context_lists = []
    for batch in pbar:
        if len(context_lists) < batch_size:
            context_lists.append(batch['context_list'])
            continue
        agent_t, partner_agent_t = [], []
        for _ in tqdm(range(max_turn_size)):
            # partner agent first
            vector, partner_agent_t_ = partner_agent.model.self_play_one_turn(context_lists, agent.vocab)
            rest = partner_searcher._search(vector, topk=100)
            rest = [random.choice(i) for i in rest]
            context_lists = [i + [j] for i,j in zip(context_lists, rest)]
            # base agent second
            vector, agent_t_ = agent.model.self_play_one_turn(context_lists, partner_agent.vocab)
            rest = searcher._search(vector, topk=100)
            rest = [random.choice(i) for i in rest]
            context_lists = [i + [j] for i,j in zip(context_lists, rest)]
            agent_t.append(agent_t_)
            partner_agent_t.append(partner_agent_t_)
            torch.cuda.empty_cache()
        agent_t1_avg.append(agent_t)
        partner_agent_t_avg.append(partner_agent_t)
        counter += 1
        if counter > max_number:
            break
        context_lists = [batch['context_list']]
    tc_4 = np.mean([sum(instance[-4:]) for instance in agent_t1_avg])
    tc_8 = np.mean([sum(instance[-8:]) for instance in agent_t1_avg])
    tc_16 = np.mean([sum(instance[-16:]) for instance in agent_t1_avg])
    tc_32 = np.mean([sum(instance[-32:]) for instance in agent_t1_avg])
    tc_64 = np.mean([sum(instance[-64:]) for instance in agent_t1_avg])
    tc_128 = np.mean([sum(instance[-128:]) for instance in agent_t1_avg])

    # partner time cost (dual-bert)
    tc_4_ = np.mean([sum(instance[-4:]) for instance in partner_agent_t_avg])
    tc_8_ = np.mean([sum(instance[-8:]) for instance in partner_agent_t_avg])
    tc_16_ = np.mean([sum(instance[-16:]) for instance in partner_agent_t_avg])
    tc_32_ = np.mean([sum(instance[-32:]) for instance in partner_agent_t_avg])
    tc_64_ = np.mean([sum(instance[-64:]) for instance in partner_agent_t_avg])
    tc_128_ = np.mean([sum(instance[-128:]) for instance in partner_agent_t_avg])
    print(f'[!] average speedup')
    print(f'  - 4 turn  : {round(tc_4_/tc_4, 4)}')
    print(f'  - 8 turn  : {round(tc_8_/tc_8, 4)}')
    print(f'  - 16 turn : {round(tc_16_/tc_16, 4)}')
    print(f'  - 32 turn : {round(tc_32_/tc_32, 4)}')
    print(f'  - 64 turn : {round(tc_64_/tc_64, 4)}')
    print(f'  - 128 turn: {round(tc_128_/tc_128, 4)}')

    tc_128 = np.mean([np.mean(instance[-50:]) for instance in agent_t1_avg])
    tc_128_ = np.mean([np.mean(instance[-50:]) for instance in partner_agent_t_avg])
    print(f'  - 128 average time cost: {round(tc_128, 4)}')
    print(f'  - 128 average time cost: {round(tc_128_, 4)}')


def main_self_play_test(**args):
    test_iter, (agent, partner_agent) = prepare_self_play_test_inference(**args)
    pbar = tqdm(test_iter)
    cost_time = []
    max_number = 1000
    counter = 0
    agent_t, partner_agent_t = [], []
    batch_size = 1
    context_lists = []
    for batch in pbar:
        if len(context_lists) < batch_size:
            context_lists.append(batch['context_list'])
            continue
        string = ''.join(context_lists[0])
        string_length = len(agent.vocab.encode(string))
        # if string_length < 320:
        #     context_lists = [batch['context_list']]
        #     continue
        try:
            # partner agent first
            vector, partner_agent_t_ = partner_agent.model.self_play_one_turn(context_lists, agent.vocab)
            # base agent second
            vector, agent_t_ = agent.model.self_play_one_turn(context_lists, partner_agent.vocab)
        except:
            context_lists = [batch['context_list']]
            continue
        agent_t.append(agent_t_)
        partner_agent_t.append(partner_agent_t_)
        counter += 1
        if counter > max_number:
            break
        context_lists = [batch['context_list']]
        torch.cuda.empty_cache()
    tc1 = np.mean(agent_t)
    tc2 = np.mean(partner_agent_t)
    print(f'[!] average speedup')
    print(f'  - agent turn  : {round(tc1*1000, 4)}')
    print(f'  - partner agent turn  : {round(tc2*1000, 4)}')


if __name__ == "__main__":
    args = vars(parser_args())
    if args['mode'] == 'recall':
        print(f'[!] Make sure that the inference script of model({args["model"]}) on dataset({args["dataset"]}) has been done.')
        main_recall(**args)
    elif args['mode'] == 'es_recall':
        main_es_recall(**args)
    elif args['mode'] == 'rerank':
        main_rerank(**args)
    elif args['mode'] == 'self_play':
        # main_self_play(**args)
        main_self_play_test(**args)
    elif args['mode'] == 'horse_human':
        main_horse_human(**args)
    elif args['mode'] == 'generation':
        main_generation(**args)
    elif args['mode'] == 'ppl':
        main_ppl(**args)
    elif args['mode'] == 'fg_rerank':
        main_rerank_fg(**args)
    elif args['mode'] == 'compare':
        main_compare(**args)
    elif args['mode'] == 'rerank_time':
        main_rerank_time(**args)
    elif args['mode'] == 'rerank_acc':
        main_acc_test(**args)
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')
