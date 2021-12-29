import requests
import json

# def SendPOST(url, port, method, params):
#     '''
#     remember to run: import http.client
#     parameters:
#         1. url: 9.91.66.241
#         2. port: 22351
#         3. method: /recall
#         4. params: json dumps string
#     '''
#
#     headers = {"Content-type": "application/json"}
#     conn = http.client.HTTPConnection(url, port)
#     conn.request('POST', method, params, headers)
#     response = conn.getresponse()
#     code = response.status
#     reason = response.reason
#     data = json.loads(response.read().decode('utf-8'))
#     conn.close()
#     return data

if __name__ == '__main__':
    while (True):
        print('\n\n ---请输入文本： ')
        input_text = input()
        params = {"segment_list":[{"str":input_text,"status":"editing"}], "topk":10, "lang":"zh"}
        headers = {"Content-type": "application/json"}
        url = 'http://9.91.66.241:8082/recall'
        # data = requests.post(url, params)
        data = requests.post(url, headers=headers, json=params)
        data = json.loads(data.text)
        for item in data['item_list'][0]['candidates']:
            print(item['similarity'] + "\t" + item['text'])




