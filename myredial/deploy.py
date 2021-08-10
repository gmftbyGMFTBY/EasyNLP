from config import *
from header import *
from flask import Flask, request, jsonify, make_response, session
from deploy import *

def create_app():
    app = Flask(__name__)

    rerank_args = load_deploy_config('rerank')
    recall_args = load_deploy_config('recall')
    pipeline_args = load_deploy_config('pipeline')
    pipeline_evaluation_args = load_deploy_config('pipeline_evaluation')
    if rerank_args['activate']:
        rerankagent = RerankAgent(rerank_args)
        print(f'[!] Rerank agent activate')
    if recall_args['activate']:
        recallagent = RecallAgent(recall_args)
        print(f'[!] Recall agent activate')
    if pipeline_args['activate']:
        pipelineagent = PipelineAgent(pipeline_args)
        print(f'[!] Pipeline agent activate')
    if pipeline_evaluation_args['activate']:
        pipelineevaluationagent = PipelineEvaluationAgent(pipeline_args)
        print(f'[!] Pipeline evaluation agent activate')
    
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
            data = request.json
            whole_size = pipelineevaluationagent.recallagent.searcher.searcher.ntotal
            responses, core_time = pipelineevaluationagent.work_full_rank_evaluation(
                data['segment_list'], whole_size=whole_size,
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
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            contexts = [i['str'] for i in data['segment_list']]
            rest = [{'context': c, 'response': r} for c, r in zip(contexts, responses)]
            result['item_list'] = rest
        else:
            result['item_list'] = None
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
            data = request.json
            responses, core_time = pipelineagent.work(data['segment_list'])
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
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            contexts = [i['str'] for i in data['segment_list']]
            rest = [{'context': c, 'response': r} for c, r in zip(contexts, responses)]
            result['item_list'] = rest
        else:
            result['item_list'] = None
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
            data = request.json
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
            data = request.json
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
            rest = [{'context': c, 'candidates': rs} for c, rs in zip(contexts, candidates)]
            result['item_list'] = rest
        else:
            result['item_list'] = None
        return jsonify(result)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    app_args = load_base_config()['deploy']
    app = create_app()
    app.run(
        host=app_args['host'], 
        port=app_args['port'],
    )
