'''
验证分析基于sementic retrival的paraphrase效果
'''
import requests
import json
import time


def sementic_retrival(text, k):
    start_time = time.time()
    resp = []
    params = {"segment_list": [{"str": text, "status": "editing"}], "topk": k, "lang": "zh"}
    headers = {"Content-type": "application/json"}
    url = 'http://9.91.66.241:8089/recall'
    # data = requests.post(url, params)
    data = requests.post(url, headers=headers, json=params)
    data = json.loads(data.text)
    for item in data['item_list'][0]['candidates']:
        resp.append((item['text'], item['similarity']))
        # print(item['similarity'] + "\t" + item['text'])
    print("cost time:", time.time()-start_time)
    return resp



if __name__ == '__main__':
    while (True):
        print('\n\n ---请输入文本： ')
        input_text = input()
        resp = sementic_retrival(input_text, 10)
        for item in resp:
            print(item[1] + "\t" + item[0])

        # params = {"segment_list":[{"str":input_text,"status":"editing"}], "topk":10, "lang":"zh"}
        # headers = {"Content-type": "application/json"}
        # url = 'http://9.91.66.241:8082/recall'
        # # data = requests.post(url, params)
        # data = requests.post(url, headers=headers, json=params)
        # data = json.loads(data.text)
        # for item in data['item_list'][0]['candidates']:
        #     print(item['similarity'] + "\t" + item['text'])

