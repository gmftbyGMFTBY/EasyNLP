'''
验证分析基于sementic retrival的paraphrase效果
'''
import requests
import json
import time
import Levenshtein

def cal_sent_overlap(sent1, sent2):
    '''
    计算两个句子的重合度
    '''
    denominator = max(len(set(sent1)), len(set(sent2)))
    numerator = len(set(sent1).intersection(set(sent2)))
    return numerator / float(denominator)

def common_str_detection(str1, str2):
    s1, s2 = str1, str2
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    maxstr = s1
    substr_maxlen = max(len(s1), len(s2))

    for sublen in range(substr_maxlen, -1, -1):
        for i in range(substr_maxlen - sublen + 1):
            if maxstr[i:i + sublen] in s2:
                return maxstr[i:i + sublen]


def sementic_retrieval(text, k):
    start_time = time.time()
    resp = []
    params = {"segment_list": [{"str": text, "status": "editing"}], "topk": k, "lang": "zh"}
    headers = {"Content-type": "application/json"}
    url = 'http://9.91.66.241:8082/recall'
    # data = requests.post(url, params)
    data = requests.post(url, headers=headers, json=params)
    data = json.loads(data.text)
    for item in data['item_list'][0]['candidates']:
        resp.append((item['text'], item['similarity']))
        # print(item['similarity'] + "\t" + item['text'])
    print("cost time:", time.time()-start_time)
    return resp

def sementic_retrieval_batch(texts, k):
    # start_time = time.time()
    resp = dict()
    request_data = []
    for text in texts:
        request_data.append({"str": text, "status": "editing"})
    params = {"segment_list": request_data, "topk": k, "lang": "zh"}
    headers = {"Content-type": "application/json"}
    url = 'http://9.91.66.241:8082/recall'
    # data = requests.post(url, params)
    data = requests.post(url, headers=headers, json=params)
    data = json.loads(data.text)
    # print(data)

    for item in data['item_list']:
        query = item['context']
        candidates = []
        for candidate in item['candidates']:
            candidates.append((candidate['text'], float(candidate['similarity'])))
        resp[query] = candidates

    # for item in data['item_list'][0]['candidates']:
    #     resp.append((item['text'], item['similarity']))
        # print(item['similarity'] + "\t" + item['text'])
    # print("cost time:", time.time()-start_time)
    return resp


def file_retrieve(filename):
    '''
    对输入文件语料进行batch语义检索
    :return:
    '''
    out = open("./retrieval_res_5k—2.txt", "w", encoding="utf-8")
    batch_data = []
    file = open(filename, encoding="utf-8")
    succ_cnt = 0
    idx = 0
    fail_cnt = 0
    candidate_cnt = 0
    while True:
        if idx % 500 == 0:
            print(idx)
        if idx > 65:
            break
        line = file.readline()
        if len(line.strip()) > 300:
            continue
        if line:
            idx += 1
            batch_data.append(line.strip())
            if idx % 64 == 0:
                start = time.time()
                resp = sementic_retrieval_batch(batch_data, 10)
                print("1time cost:", time.time() - start)
                for query, candidates in resp.items():
                    if candidates[1][1] >= 100:
                        fail_cnt += 1
                    c_id = 1
                    while c_id < len(candidates) and candidates[c_id][1] < 100:
                        retrieve_text, similarity = candidates[c_id][0], candidates[c_id][1]
                        if similarity == 0:
                            c_id += 1
                            continue

                        min_sen = min(len(query), len(retrieve_text))
                        max_sen = max(len(query), len(retrieve_text))

                        overlap = cal_sent_overlap(query, retrieve_text)

                        common_str = common_str_detection(query, retrieve_text)
                        common_str_ratio = round(len(common_str) / min_sen, 2)

                        dis = Levenshtein.distance(query, retrieve_text)
                        levenshtein_ratio = round(dis / max_sen, 2)

                        candidate_cnt += 1

                        # 抛弃句子字符重合率大于85%的句子对
                        if overlap < 0.85 and common_str_ratio < 0.7 and levenshtein_ratio > 0.25 and levenshtein_ratio < 0.9:
                            out.write(str(similarity) + "\t" + str(round(overlap, 2)) + "\t" + str(common_str_ratio) + "\t" + str(levenshtein_ratio) + "\t" + query + "\t" + retrieve_text + "\n")
                            succ_cnt += 1
                        c_id += 1
                    # if candidates[1][1] < 100 and candidates[1][1] != 0:
                    #     overlap = cal_sent_overlap(query, candidates[1][0])
                    #     #抛弃句子字符重合率大于85%的句子对
                    #     if overlap < 0.85:
                    #         out.write(candidates[1][1] + "\t" + str(round(overlap, 2)) + "\t" + query + "\t" + candidates[1][0] + "\n")
                    #         succ_cnt += 1
                batch_data = []
        else:
            break
    print("total:{}  success:{}  fail:{}  candidate cnt:{}".format(idx, succ_cnt, fail_cnt, candidate_cnt))
    out.close()


if __name__ == '__main__':
    # while (True):
    #     print('\n\n ---请输入文本： ')
    #     input_text = input()
    #     text_batch = []
    #     for i in range(64):
    #         text_batch.append(input_text + str(i))
    #     # resp = sementic_retrival(input_text, 10)
    #     resp = sementic_retrieval_batch(text_batch, 10)
    #
    #     for item in resp:
    #         print(item[1] + "\t" + item[0])
    start_time = time.time()
    filename = "/apdcephfs/share_916081/willzychen/wz_data/esports_v1.txt"
    file_retrieve(filename)
    print("cost time:", time.time()-start_time)

