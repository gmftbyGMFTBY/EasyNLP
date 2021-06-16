from model import *
from config import *
from header import *
from flask import Flask, request, jsonify, make_response, session
from deploy import *

def create_app():
    app = Flask(__name__)

    rerank_args = load_deploy_config('rerank')
    recall_args = load_deploy_config('recall')
    rerankagent = RerankAgent(rerank_args)
    recallagent = RecallAgent(recall_args)

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
        time_begin = time.time()
        core_time_begin = time.time()
        try:
            data = request.json
            rest = rerankagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            print(error)
            succ = False
        core_time_end = time.time()
        time_end = time.time()


        # packup
        result = {
            'header': {
                'time_cost_ms': 1000 * (time_end - time_begin),
                'time_cost': time_end - time_begin,
                'core_time_cost_ms': 1000 * (core_time_end - core_time_begin),
                'core_time_cost': core_time_end - core_time_begin,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
            rest_ = []
            for scores, batch in zip(rest, data['segment_list']):
                item = {'context': batch['context']}
                item['candidates'] = []
                for s, cand in zip(scores, batch['candidates']):
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
                    'candidates': [ 
                        'candidates1',
                        ...
                    ]
                }
            ]
        }
        '''
        time_begin = time.time()
        core_time_begin = time.time()
        try:
            data = request.json
            candidates = recallagent.work(data['segment_list'])
            succ = True
        except Exception as error:
            print(error)
            succ = False
        core_time_end = time.time()
        time_end = time.time()

        # packup
        result = {
            'header': {
                'time_cost_ms': 1000 * (time_end - time_begin),
                'time_cost': time_end - time_begin,
                'core_time_cost_ms': 1000 * (core_time_end - core_time_begin),
                'core_time_cost': core_time_end - core_time_begin,
                'ret_code': 'succ' if succ else 'fail',
            }, 
        }
        if succ:
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
