# Easy-to-use toolkit for retrieval-based Chatbot

## Note

- [x] The rerank performance of dual-bert-fusion is bad, the reason maybe that the context information is only for ground-truth, but the other negative samples lost their corresponding context, and during rerank procedure, we use the context of the ground-truth for all the candidates, which may pertubate the decison of the model.
- [ ] test the simcse for the only one conversation context utterance, q-q matching (context similarity)
- [x] Generate the gray data need the faiss Flat  index runnign on GPU, which only costs 6~7mins for 0.5 million dataset
- [ ] implement UMS-BERT and BERT-SL using the post-train checkpoint of the BERT-FP
- [ ] implement my own post train procedure
- [ ] implement R-Drop for bert-ft (add the similarity on the cls embedding) and dual-bert
- [x] fix the bugs of _length_limit of the bert-ft
- [ ] dynamic margin (consider the margin between the original rerank scores)
- [ ] comparison: bce to three-classification(positive, negative, hard to tell); hard to tell could use the self-comparison and the comparison with the top-1 retrieved result

## How to Use

1. Init the repo
Before using the repo, please run the following command to init:

```bash
# create the necessay folders
python init.py

# prepare the environment
# if some package cannot be installed, just google and install it from other ways
pip install -r requirements.txt
```

2. train the model

```bash
./scripts/train.sh <dataset_name> <model_name> <cuda_ids>
# or, noted that start.sh read the config from jizhi_config.json to start the training task
./start.sh 
```

3. test the model [rerank]

```bash
./scripts/test_rerank.sh <dataset_name> <model_name> <cuda_id>
```

4. test the model [recal]

```bash
# different recall_modes are available: q-q, q-r
./scripts/test_recall.sh <dataset_name> <model_name> <cuda_id>
```

5. inference the responses and save into the faiss index

Somethings inference will missing data samples, please use the 1 gpu (faiss-gpu search use 1 gpu quickly)

It should be noted that:
    1. For writer dataset, use `extract_inference.py` script to generate the inference.txt
    2. For other datasets(douban, ecommerce, ubuntu), just `cp train.txt inference.txt`. The dataloader will automatically read the test.txt to supply the corpus. 

```bash
# work_mode=response, inference the response and save into faiss (for q-r matching) [dual-bert/dual-bert-fusion]
# work_mode=context, inference the context to do q-q matching
# work_mode=gray, inference the context; read the faiss(work_mode=response), search the topk hard negative samples; remember to set the BERTDualInferenceContextDataloader in config/base.yaml
./scripts/inference.sh <dataset_name> <model_name> <cuda_ids>
```

If you want to generate the gray dataset for the dataset:

```bash
# 1. set the mode as the **response**, to generate the response faiss index; corresponding dataset name: BERTDualInferenceDataset
./scripts/inference.sh <dataset_name> <model_name> <cuda_ids>

# 2. set the mode as the **gray**, to inference the context in the train.txt and search the top-k candidates as the gray(hard negative) samples; corresponding dataset name: BERTDualInferenceContextDataset
./scripts/inference.sh <dataset_name> <model_name> <cuda_ids>
```

6. deploy the rerank and recall model

```bash
# load the model on the cuda:0(can be changed in deploy.sh script)
./scripts/deploy.sh <cuda_id>
```
at the same time, you can test the deployed model by using:

```bash
# test_mode: recall, rerank, pipeline
./scripts/test_api.sh <test_mode> <dataset>
```

7. jizhi start (just for tencent)

```bash
# need to edit the configuration in jizhi_config.json
./jizhi_start.sh
```

8. test the recall performance of the elasticsearch

Before testing the es recall, make sure the es index has been built:
```bash
# recall_mode: q-q/q-r
./scripts/build_es_index.sh <dataset_name> <recall_mode>
```

```bash
# recall_mode: q-q/q-r
./scripts/test_es_recall.sh <dataset_name> <recall_mode> 0
```

## Ready models and datasets

### 1. Models
1. bert-ft
2. dual-bert
3. dual-bert-cl
4. dual-bert-cl-gate
5. poly-encoder
6. dual-bert-gray-writer 
7. hash-bert
8. sa-bert

### 2. Datasets
1. E-Commerce
2. Douban
3. Ubuntu-v1
4. Writer (智能写作助手数据集)
5. LCCC
