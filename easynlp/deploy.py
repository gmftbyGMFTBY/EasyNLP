from config import *
from header import *
from flask import Flask, request, jsonify, make_response, session
from deploy import *
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--base_port', type=int, default=22330)
    return vars(parser.parse_args())

def create_app():
    app = Flask(__name__)

    rerank_args = load_deploy_config('rerank')
    generation_args = load_deploy_config('generation')
    generation_dialog_args = load_deploy_config('generation_dialog')
    recall_args = load_deploy_config('recall')
    pipeline_args = load_deploy_config('pipeline')
    pipeline_evaluation_args = load_deploy_config('pipeline_evaluation')
    evaluation_args = load_deploy_config('evaluation')
    if rerank_args['activate']:
        rerankagent = RerankAgent(rerank_args)
        print(f'[!] Rerank agent activate')
        rerank_logger = init_logging(rerank_args)
    if generation_dialog_args['activate']:
        generationdialogagent = DeployGenerationDialogAgent(generation_dialog_args)
        print(f'[!] Generation Dialog agent activate')
        generation_dialog_logger = init_logging(generation_dialog_args)
    if generation_args['activate']:
        generationagent = DeployGenerationAgent(generation_args)
        print(f'[!] Generation agent activate')
        generation_logger = init_logging(generation_args)
    if recall_args['activate']:
        recallagent = RecallAgent(recall_args)
        print(f'[!] Recall agent activate')
        recall_logger = init_logging(recall_args)
    if pipeline_args['activate']:
        pipelineagent = PipelineAgent(pipeline_args)
        print(f'[!] Pipeline agent activate')
        pipeline_logger = init_logging(pipeline_args, pipeline=True)
    if pipeline_evaluation_args['activate']:
        pipelineevaluationagent = PipelineEvaluationAgent(pipeline_evaluation_args)
        print(f'[!] Pipeline evaluation agent activate')
        pipeline_evaluation_logger = init_logging(pipeline_evaluation_args, pipeline=True)
    if evaluation_args['activate']:
        evaluationagent = DeployEvaluationAgent(evaluation_args)
        print(f'[!] Evaluation evaluation agent activate')
        evaluation_logger = init_logging(evaluation_args)
    
    @app.route('/pipeline_evaluation', methods=['POST'])
    def pipeline_evaluation_api():
        '''
        {
            'segment_list': [
                {'str': 'context sentence1', 'status': 'editing'},
                ...
            ]
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }

        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': 'context sentence1',
                    'response': 'candidates1',
                }
            ]
        }
        '''
        try:
            data = json.loads(request.data)
            # (responses, mrrs, recall_t, rerank_t), core_time = pipelineevaluationagent.work(
            (responses, recall_t, rerank_t), core_time = pipelineevaluationagent.work(
                data['segment_list'],
                topk=pipeline_evaluation_args['recall']['topk'],
            )
            succ = True
        except Exception as error:
            core_time = 0
            print('ERROR:', error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'recall_core_time_cost_ms': 1000 * recall_t,
                'rerank_core_time_cost_ms': 1000 * rerank_t,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            contexts = [i['str'] for i in data['segment_list']]
            # rest = [{'context': c, 'response': r, 'mrr': mrr} for c, r, mrr in zip(contexts, responses, mrrs)]
            rest = [{'context': c, 'response': r} for c, r in zip(contexts, responses)]
            result['item_list'] = rest
            result['results'] = {}
            # show the evaluation results
            for name in ['R@1000', 'R@500', 'R@100', 'R@50', 'MRR']:
                value = round(np.mean(pipelineevaluationagent.collection[name]), 4)
                result['results'][name] = value
        else:
            result['item_list'] = None

        # log
        push_to_log(result, pipeline_evaluation_logger)

        return jsonify(result)
    
    @app.route('/pipeline', methods=['POST'])
    def pipeline_api():
        '''
        {
            'segment_list': [
                {'str': 'context sentence1', 'status': 'editing'},
                ...
            ]
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }

        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': 'context sentence1',
                    'response': 'candidates1',
                }
            ]
        }
        '''
        try:
            data = json.loads(request.data)
            (responses, recall_t, rerank_t), core_time = pipelineagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            core_time = 0
            print('ERROR:', error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'recall_core_time': recall_t,
                'rerank_core_time': rerank_t,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            contexts = [i['str'] for i in data['segment_list']]
            rest = [{'context': c, 'response': r} for c, r in zip(contexts, responses)]
            result['item_list'] = rest
        else:
            result['item_list'] = None
        # log
        push_to_log(result, pipeline_logger)
        return jsonify(result)

    @app.route('/rerank', methods=['POST'])
    def rerank_api():
        '''
        {
            'segment_list': [
                {
                    'context': 'context1', 
                    'candidates': [
                        'candidates1-1',
                        'candidates1-2',
                        ...
                    ]
                    'status': 'editing'
                },
                ...
            ],
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }
        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': 'context sentence1',
                    'candidates': [ 
                        {'str': 'candidates1', 'score': 0.5},
                        ...
                    ]
                }
            ]
        }
        '''
        try:
            # data = request.json
            data = json.loads(request.data)
            rest, core_time = rerankagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            core_time = 0
            print(error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            rest_ = []
            for scores, batch in zip(rest, data['segment_list']):
                item = {'context': batch['context']}
                item['candidates'] = []
                for s, cand in zip(scores, batch['candidates']):
                    if rerank_args['model'] in ['gpt2lm', 'kenlm']:
                        item['candidates'].append({'str': cand, 'score': s[1], 'ppl': s[0]})
                    else:
                        item['candidates'].append({'str': cand, 'score': s})
                rest_.append(item)
            result['item_list'] = rest_
        else:
            result['item_list'] = None
        # log
        push_to_log(result, rerank_logger)
        return jsonify(result)

    @app.route('/recall', methods=['POST'])
    def recall_api():
        '''
        {
            'segment_list': [
                {'str': 'context sentence1', 'status': 'editing'},
                ...
            ],
            # topk is optinal, if topk key doesn't exist, default topk will be used (100)
            'topk': 100,
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }

        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': 'context sentence1',
                    'candidates': [ 
                        {
                            'context': 'context sentence1',
                            'candidates1': {
                                'text': 'candidate sentence', 
                                'source': {'title': 'title', 'url': 'url'}
                            }
                        },
                        ...
                    ]
                }
            ]
        }
        '''
        try:
            data = json.loads(request.data)
            topk = data['topk'] if 'topk' in data else None
            candidates, core_time = recallagent.work(data['segment_list'], topk=topk)
            succ = True
        except Exception as error:
            core_time = 0
            print(error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            contexts = [i['str'] for i in data['segment_list']]
            ground_truths = [i['ground_truth'] for i in data['segment_list']]
            rest = [{'context': c, 'candidates': rs, 'ground_truth': g} for g, c, rs in zip(ground_truths, contexts, candidates)]
            result['item_list'] = rest
        else:
            result['item_list'] = None
        # log
        #push_to_log(result, recall_logger)
        return jsonify(result)

    @app.route('/evaluation', methods=['POST'])
    def evaluation_api():
        '''
        {
            'segment_list': [
                {
                    'context': 'context1', 
                    'candidate1': 'candidate1',
                    'candidate2': 'candidate2',
                    'status': 'editing'
                },
                ...
            ],
        }
        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': ['context sentence1'],
                    'candidate1': 'candidate1',
                    'candidate2': 'candidate2',
                    'score': advantage score of candidate1 over candidate2
                }
            ]
        }
        '''
        try:
            # data = request.json
            data = json.loads(request.data)
            item_list, core_time = evaluationagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            item_list = []
            core_time = 0
            print(error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        result['item_list'] = item_list
        push_to_log(result, evaluation_logger)
        return jsonify(result)

    @app.route('/generation', methods=['POST'])
    def generation_api():
        '''
        {
            'segment_list': [
                {
                    'context': 'context1', 
                    'status': 'editing'
                },
                ...
            ],
            'decoding_method': 'contrastive_batch_search',    # defualt is the contrastive_search_batch
            'generation_num': 3,    # default generation_num is 3
            'max_gen_len': 64,
            'sampling_prefix_len': 5,
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }
        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': 'context sentence1',
                    'contrastive_search_reference': [ 
                        {'str': 'generation_result_1'},
                        ...
                    ],
                    'contrastive_search_diverse': [ 
                        {'str': 'generation_result_1'},
                        ...
                    ],
                    'topk_topp_sampling': [ 
                        {'str': 'generation_result_1'},
                        ...
                    ],
                }
            ]
        }
        '''
        try:
            # data = request.json
            data = json.loads(request.data)
            (g1, g2, g3), core_time = generationagent.work(data)
            succ = True
        except Exception as error:
            core_time = 0
            print(error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            rest_ = []
            for g1_, g2_, g3_, batch in zip(g1, g2, g3, data['segment_list']):
                item = {'context': batch['context']}
                item['contrastive_search_reference'] = [{'str': s} for s in g2_]
                item['contrastive_search_diverse'] = [{'str': s} for s in g1_]
                item['topk_topp_sampling'] = [{'str': s} for s in g3_]
                rest_.append(item)
            result['item_list'] = rest_
        else:
            result['item_list'] = None
        push_to_log(result, generation_logger)
        return jsonify(result)

    @app.route('/generation_dialog', methods=['POST'])
    def generation_dialog_api():
        '''
        {
            'segment_list': [
                {
                    'context': ['context1', 'context2', ...], 
                    'status': 'editing'
                },
                ...
            ],
            'lang': 'zh',
            'uuid': '',
            'user': '',
        }
        {
            'header': {
                'time_cost_ms': 0.01,
                'time_cost': 0.01,
                'core_time_cost_ms': 0.01,
                'core_time_cost': 0.01,
                'ret_code': 'succ'
            },
            'item_list': [
                {
                    'context': ['context1', 'context2', ...], 
                    'response': 'response1',
                }
            ]
        }
        '''
        try:
            # data = request.json
            data = json.loads(request.data)
            g, core_time = generationdialogagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            core_time = 0
            print(error)
            succ = False

        # packup
        result = {
            'header': {
                'core_time_cost_ms': 1000 * core_time,
                'core_time_cost': core_time,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            rest_ = []
            for g_, batch in zip(g, data['segment_list']):
                item = {'context': batch['context']}
                item['response'] = g_
                rest_.append(item)
            result['item_list'] = rest_
        else:
            result['item_list'] = None
        push_to_log(result, generation_dialog_logger)
        return jsonify(result)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # mentioned !
    # base_port = parser_args()['base_port']

    app_args = load_base_config()['deploy']
    # print(f'[!] running port: {base_port+app_args["port"]}')
    app = create_app()
    app.run(
        host=app_args['host'], 
        port=app_args['port'],
        # port=base_port+app_args['port'],
    )
