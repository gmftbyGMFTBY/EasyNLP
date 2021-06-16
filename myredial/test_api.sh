#!/bin/bash

# test recall api
# curl -H "Content-Type: application/json" -X POST -d '{"segment_list": [{"str": "今天的天气不错", "status": "editing"}], "lang": "zh"}'  http://9.91.66.241:22335/recall

# test rerank api
curl -H "Content-Type: application/json" -X POST -d '{"segment_list": [{"context": "今天的天气不错", "candidates": ["是啊，挺风和日丽的", "适合去野外踏青", "我今天要加班"], "status": "editing"}], "lang": "zh"}'  http://9.91.66.241:22335/rerank
