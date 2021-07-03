# Easy-to-use toolkit for retrieval-based Chatbot

## Note

- [x] The rerank performance of dual-bert-fusion is bad, the reason maybe that the context information is only for ground-truth, but the other negative samples lost their corresponding context, and during rerank procedure, we use the context of the ground-truth for all the candidates, which may pertubate the decison of the model.
- [ ] test the simcse for the only one conversation context utterance, q-q matching (context similarity)
- [x] Generate the gray data need the faiss Flat  index runnign on GPU, which only costs 6~7mins for 0.5 million dataset
- [ ] implement UMS-BERT and BERT-SL using the post-train checkpoint of the BERT-FP
- [ ] implement my own post train procedure
- [ ] implement R-Drop for bert-ft and dual-bert
- [x] fix the bugs of _length_limit of the bert-ft

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

It should be noted that:
1. For writer dataset, use `extract_inference.py` script to generate the inference.txt
2. For other datasets(douban, ecommerce, ubuntu), just `cp train.txt inference.txt`. The dataloader will automatically read the test.txt to supply the corpus. 

```bash
# work_mode=response, inference the response and save into faiss (for q-r matching) [dual-bert/dual-bert-fusion]
# work_mode=context, inference the context to do q-q matching
# work_mode=gray, inference the context; read the faiss(work_mode=response), search the topk hard negative samples; remember to set the BERTDualInferenceContextDataloader in config/base.yaml
./inference <dataset_name> <model_name> <cuda_ids>
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


## Experiment Results
### 1. E-Commerce Dataset

* recall performance

| Model (CPU/109105)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat(q-r)    |  0.567 | 0.791   | 29.51               |
| dual-bert-fusion(q-r)  |  0.561 | 0.645   | -              |

* rerank performance

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| BERT-FP        | 87.0  | 95.6  | 99.3  | -      |
| dual-bert      | 91.7  | 96.0  | 99.2  | 94.85  |
| dual-bert-fusion | 84.7  | 93.9 | 98.7  | 90.0  |


| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| bert-ft(dup=5) | 81.4  | 94.1  | 99.1  | 89.37  |
| dual-bert-hier(bsz=64, epoch=10, shuffle-ddp) | 88.8 | 95.8  | 98.6 | 93.32 |
| dual-bert-hier(bsz=128, epoch=10, shuffle-ddp) | 90.7 | 96.5  | 99.3 | 94.5 |
| dual-bert-hier(bsz=64, epoch=5, bert-post) | 87.0 | 93.8 | 98.7 | 91.97 |
| dual-bert-hier(bsz=64, epoch=5, bert-dual-post) | 89.4 | 95.2 | 99.2 | 93.64 |
| dual-gru(bsz=32, epoch=5) | 71.3 | 85.7 | 96.2 | 81.98 |
| dual-gru(bsz=64, epoch=5) | 73.6 | 86.3 | 96.4 | 83.22 |
| dual-gru(bsz=128, epoch=5) | 72.4 | 85.3 | 95.5 | 82.44 |
| dual-gru(bsz=256, epoch=5) | 67.1 | 82.3 | 94.8 | 78.97 |
| dual-gru-no-gru(bsz=64, epoch=5) | 69.3 | 83.8 | 93.5 | 80.51 |
| dual-gru(bsz=64, epoch=5, last-utterance) | 73.8 | 86.7 | 96.7 | 83.58 |
| dual-bert-scm(bsz=16, epoch=5, bert-post) | 86.1 | 94.3 | 99.1 | 91.72 |
| dual-bert-scm(bsz=16, epoch=5, bert-post) | 86.0 | 93.5 | 98.4 | 91.44 |
| dual-bert-scm(bsz=16, epoch=5, bert-post, multi-head) | 86.2 | 94.1 | 99.1 | 91.73 |
| dual-bert(bsz=16, epoch=5, bert-post) | 86.1 | 94.1 | 99.2 | 91.71 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c) | 88.6 | 94.5 | 99.1 | 93.04 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c/c-c/r-r) | 87.2 | 94.8 | 99.3 | 92.37 |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c, single) | 85.4 | 94.0 | 99.0 | 91.25 |
| dual-bert-poly(bsz=16, epoch=5, bert-post) | 85.3 | 94.5 | 98.9 | 91.32 |
| dual-bert-adv(bsz=16, epoch=5, bert-post) | | | | |
| dual-bert-adv(bsz=16, epoch=10, shuffle-ddp) | 87.4 | 94.2 | 98.8 | 92.3 |
| dual-bert-cl(bsz=16) | 87.5  | 94.4  | 99.1  | 92.48 |
| dual-bert-cross(bsz=16, node=7) | 85.0  | 92.6  | 97.4  | 90.57 |

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder(bsz=16) | 80.8  | 91.7  | 98.3  | 88.39  |
| Bi-Encoder(bsz=16, compare with dual-bert-cl) | 78.7  | 90.8  | 97.9  | 87.1  |
| Bi-Encoder-cl(bsz=16) | 78.3  | 90.7  | 97.7  | 86.8  |
| Bi-Encoder-mb(bsz=16, mb=4096) | 80.6  | 91.0  | 97.9  | 88.09  |
| Bi-Encoder(bsz=64) | 83.7  | 92.4  | 98.5  | 90.02  |
| Bi-Encoder-one2many(bsz=16) | 90.4  | 95.6  | 98.9  | 94.16  |
| Bi-Encoder-one2many(bsz=16, bert-post, single head) | 91.7 | 95.4  | 99.4  | 94.83  |
| Bi-Encoder-one2many-bad(bsz=16,head=5,max,pre-extract=50) | 91.0  | 95.5  | 99.3  | 94.46  |
| Bi-Encoder-one2many-pseudo(bsz=16,head=5,max,pre-extract=50) | 89.4  | 95.2  | 99.2  | 93.5  |
| Bi-Encoder-one2many-concat(bsz=16,head=5,max,pre-extract=50) | 89.5  | 94.4  | 98.8  | 93.42  |
| Bi-Encoder-one2many-no-additional-loss(bsz=16,head=5,max) | 75.5  | 89.1  | 98.2  | 85.28  |
| Bi-Encoder-one2many(bsz=16,head=5,mean,pre-extract=10) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 86.1  | 92.9  | 98.5  | 91.3  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 88.2  | 94.7  | 99.2  | 92.85  |

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| Bi-Encoder(bsz=16) | 79.7  | 90.8  | 98.3  | 87.63  |
| Bi-Encoder-one2many(bsz=16,head=5,mean) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-one2many-bad(bsz=16,head=5,max) | 91.0  | 95.5  | 99.3  | 94.46  |
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 86.1  | 92.9  | 98.5  | 91.3  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 88.2  | 94.7  | 99.2  | 92.85  |
| Bi-Encoder-hier(bsz=16) | 80.5  | 91.1  | 98.5  | 88.14  |
| Bi-Encoder-hier-multi(bsz=16,m=5) | 80.6  | 91.6  | 98.6  | 88.01  |
| Bi-Encoder(bsz=64) | 83.7  | 92.4  | 98.5  | 90.02  |
| Bi-Encoder-CL2 | 80.6  | 91.1  | 98.2  | 88.16  |
| Poly-Encoder   | 80.0  | 90.3  | 97.9  | 87.69  |
| BERT-FT        | 62.3  | 84.2  | 98    | 77.59  |
| BERT-FT        | 60.8  | 82.8  | 97.2  | 76.33  |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |
| BERT-Gen-FT    | 63.3  | 83.5  | 97.1  | 77.71  |
| BERT-Gen-FT w/o Gen | | | | |

### 2. Douban Dataset

* Recall Performance

| Model (CPU/684208)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-cl-flat      | 0.1124 | 0.2129  | 204.85              |
| dual-bert-cl-LSH       | 0.099  | 0.1994  | 13.51               |
| dual-bert-flat         | 0.1079 | 0.1919  | 197.65              |
| dual-bert-IVF8192,Flat | 0.057  | 0.0795  | 16.91               |
| dual-bert-IVF100,Flat  | 0.0675 | 0.1199  | 29.62               |
| dual-bert-LSH          | 0.0825 | 0.1723  | 12.05               |
| hash-bert-flat         | 0.045  | 0.1109  | 13.43               |
| hash-bert-BHash512     | 0.0435 | 0.1064  | 7.26                |

<!-- 
It should be noted that the difference between q-q and q-r is the index.
If the index is based on the responses, it is q-r matching;
If the index is based on the contexts, it is q-q matching

q-q: the constrastive loss is used for context context pair;
q-r: the constrastive loss is used for context response pair
-->
| Model (CPU/442787)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat(q-r)    | 0.1229 | 0.2339  | 124.06              |
| dual-bert-fusion(q-r, ctx=full)  | 0.7826 | 0.8111  | 170.83              |
| dual-bert-fusion(q-r, ctx=1)  | 0.6087  | 0.6837 | 172.2              |
| ES(q-q) | 1.0 | 1.0 | 123.19 |
| ES(q-r) | 0.1034 | 0.1514 | 39.95 |

<!--the influence of the number of the negative samples, max_len=256/64
It can be found that the number of the negative samples has the limited performance gain
it can also be found that the hard negative samples seems has the limited performance gain on the dual-encoder model
BERT-FP的post-train checkpoint和他的数据并不能共同的提高效果，分别使用可以有提升。可能是这两种的目标是一致的
-->
| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| dual-bert          | 30.42 | 50.61 | 82.6  | 65.64 | 48.28 | 61.46  |
| hash-bert-poly(head=10)    | 25.67 | 42.65 | 75.36 | 59.78 | 41.68 | 54.83  |
| hash-bert-poly(head=5)     | 25.96 | 43.24 | 75.01 | 59.64 | 41.23 | 55.15  |
| hash-bert-256      | 26.49 | 44.53 | 77.27 | 60.79 | 42.58 | 56.63  |
| hash-bert-128-(q_alpha=0.1)| 27.81 | 44.47 | 78.17 | 61.6 | 44.23 | 57.25  |
| hash-bert-128      | 27.9  | 45.01 | 77.44 | 61.95 | 44.23 | 57.55  |
| hash-bert-128-hsz(1024) | 27.3 | 44.15 | 77.54 | 61.3 | 43.63 | 57.04  |
| hash-bert-64-kl-div| 28.7 | 44.75 | 77.09 | 62.11 | 44.83 | 57.67 |
| hash-bert-64-kl-div(scale=50)| 27.85 | 45.71 | 78.18 | 62.04 | 43.93 | 57.65 |
| hash-bert-64-ft(11)| 27.37 | 43.19 | 78.31 | 61.24 | 43.93 | 56.48  |
| hash-bert-128-epoch10 | 27.6 | 45.51 | 77.79 | 61.58 | 43.48 | 57.33|
| hash-bert-128-5e-3-epoch10 | 26.57 | 42.11 | 73.56 | 59.26 | 41.68 |55.11  |
| hash-bert-128-tanh | 26.62 | 44.56 | 76.67 | 60.81 | 42.57 | 56.64  |
| hash-bert-ft(5e-4) | 26.19 | 44.63 | 76.49 | 60.35 | 42.13 | 56     |
| hash-bert-ft(5e-5) | 27.89 | 43.53 | 76.54 | 61.69 | 44.83 | 56.8   |

* Rerank performance

| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP(512)       | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| bert-ft(320, bert-fp)    | 29.63 | 50.95 | 86.3 | 66.07 | 48.13 | 61.76  |
| bert-ft+compare(320, bert-fp, margin=0.55)    | 30.38 | 50.63 | 86.2 | 66.62 | 49.03 | 62.34  |
| bert-ft+compare(320, bert-fp, margin=0.5)    | 29.43 | 51.37 | 87.08 | 66.2 | 47.83 | 62.17  |
| poly encoder(poly-m=32)  | 31.76 | 50.59 | 85.72 | 66.49 | 49.48 | 62.84  |
| dual-bert(bsz=32, epoch=5, poly-m=32) | 30.55 | 46.93 | 81.16 | 64.45 | 48.43  | 60.34 |
| dual-bert(bsz=32, epoch=5, bert-fp) | 31.42 | 51.6 | 83.46 | 66.41 | 49.48  | 62.22 |
| dual-bert(bsz=48, epoch=5, bert-fp) | 31.63 | 51.22 | 83.23 | 66.47 | 49.78  | 62.22 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=1) | 31.0 | 54.47 | 85.85 | 67.51 | 49.63  | 63.42 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2) | 31.0 | 54.16 | 86.28 | 67.52 | 49.63  | 63.54 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2, margin=0.55) | 31.23 | 54.41 | 86.2 | 67.74 | 49.93  | 63.58 |
| dual-bert-compare(bsz=48, epoch=5, bert-fp) | 31.05 | 49.63 | 83.84 | 65.54 | 47.53  | 61.74 |
| dual-bert-compare(loss1+loss2+loss3, bsz=32, epoch=5, gray_num=10, bert-fp) | 30.42 | 50.38 | 82.38 | 65.3 | 48.13  | 61.11 |
| dual-bert-compare(loss1+loss2+loss3, bsz=32, epoch=5, gray_num=5, bert-fp) | 29.04 |48.55 | 83.05 | 64.13 | 46.48  | 60.19 |
| dual-bert-fusion(bsz=48, epoch=5, bert-fp) | 27.95 | 48.27 | 81.93 | 63.75 | 45.58 | 59.66 |
| dual-bert-cl(bsz=48, epoch=5, bert-fp) | 30.25 | 50.69| 82.44 | 65.49 | 48.13  | 61.25 |
| dual-bert-cl(bsz=32, epoch=5, bert-fp) |  | | |  |  | |
| dual-bert-cl2(bsz=48, epoch=5, bert-fp) |  | | |  |  | |
| dual-bert-cb(bsz=32, epoch=5, bert-fp) | 31.29 | 49.45 | 81.51 | 65.92 | 49.18  | 61.39 |
| dual-bert(bsz=32, epoch=5, bert-fp, proj_dim=1024) | 31.57 | 51.67 | 83.41 | 66.48 | 49.63  | 62.26 |
| dual-bert(bsz=32, epoch=5, bert-fp, lambda(gen)=0.1) | 30.95 | 50.65 | 82.95 | 65.98 | 49.03  | 61.82 |
| dual-bert-gen(bsz=32, epoch=10) | 29.68 | 47.08 | 79.65 | 63.6 | 46.78  | 59.37 |
| dual-bert-gen(bsz=32, epoch=10, bert-fp) | 29.51 | 48.64 | 81.13 | 64.13 | 46.63  | 60.1 |
| dual-bert-gen(bsz=32, epoch=5, bert-fp) | 29.48 | 49.69 | 80.74 | 64.31 | 46.48  | 60.24 |
| dual-bert-gen(bsz=32, epoch=5) | |  | |  |  | |
| dual-bert-gen(bsz=32, epoch=5, bert-fp) | | | | | | |
| dual-bert(bsz=16, epoch=5, bert-post) | 27.85 | 49.26 | 85.99 | 63.88 | 44.83 | 60.73 |
| dual-bert(bsz=16, epoch=5, bert-post, extra_t=16) | 30.26 | 51.2 | 85.71 | 65.93 | 47.98 | 62.22  |
| dual-bert(bsz=16, epoch=5, bert-post, extra_t=32) | 31.36 | 51.32 | 85.82 | 66.63 | 49.33 | 62.91 |
| dual-bert(bsz=16, epoch=5, bert-post, extra_t=48) | 32.01 | 50.65 | 85.07 | 66.8 | 49.78 | 62.86 |
| dual-bert(bsz=16, epoch=5, bert-post, extra_t=64) | 30.13 | 51.27 | 85.2 | 65.72 | 47.68 | 61.92 |
| dual-bert-bm25(bsz=16, epoch=5, bert-post, head_num=5) | 31.18 | 51.36 | 84.4 | 66.56 | 48.88 | 62.2 |
| dual-bert(bsz=32, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 32.57 | 52.33 | 84.37 | 67.47 | 50.82 | 63.13 |
| dual-bert(bsz=96, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 30.65 | 51.13 | 83.05 | 66.02 | 48.88 | 61.81 |
| dual-bert-ma(bsz=32, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 32.12 | 51.63 | 82.93 | 66.86 | 50.37 | 62.54 |
| dual-bert-poly(bsz=32, epoch=5, bert-fp-post, max_len=256/64) | 31.76 | 53.25 | 84.75 | 67.51 | 50.67 | 63.27 |



| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| dual-bert(bsz=16, epoch=5, bert-post) | 30.15 | 49.42  | 84.51 | 65.65 | 48.28 | 61.62  |
| dual-bert(bsz=32, epoch=5, bert-post) | 31.01 | 51.59  | 84.68 | 66.29 | 48.88 | 62.49  |
| dual-bert(bsz=32, epoch=5, bert-post, speaker for context encoder, max-res-len=256) | 30.91 | 50.57  | 84.82 | 66.19 | 48.58 | 62.36  |
| dual-bert(bsz=48, epoch=5, bert-post, speaker for context encoder, max-res-len=256, lr=1e-4) | 30.91 | 50.57  | 84.82 | 66.19 | 48.58 | 62.36  |
| dual-bert(bsz=32, epoch=5, bert-post, speaker for context encoder, max-res-len=64) | 30.45 | 50.78  | 84.91 | 65.82 | 48.28 | 62.13 |
| dual-bert(bsz=32, epoch=5, bert-post, speaker for context encoder, max-len=512) | | | | | | |
| dual-bert(bsz=16, epoch=5, bert-post, extend-negative) | 31.00 | 49.73  | 83.91 | 65.92 | 47.83 | 61.97  |
| dual-gru(bsz=32, epoch=5) | 20.38 | 36.67  | 73.52 | 54.59 | 34.78 | 50.41  |
| dual-bert(bsz=16, epoch=5, bert-post, c-r/r-c) | 30.08 | 50.69  | 84.04 | 65.49 | 47.98 | 61.63  |
| dual-bert-hier(bsz=32, epoch=5, bert-post) | 29.1 | 49.22 | 82.16 | 64.73 | 47.23 | 60.5 |
| dual-bert-hier(bsz=32, epoch=5, bert-dual-post) | 29.74 | 47.43 | 82.06 | 64.83 | 47.83 | 60.6 |
| dual-bert-hier(bsz=32, epoch=5, bert-dual-post, mean) | 28.79 | 47.49 | 82.43 | 63.9 | 46.48 | 59.84 |
| dual-bert-hier-trs(bsz=128, epoch=5, bert-post, bert-trs-gru, c-r/r-c)| 29.4 | 48.54 | 82.16 | 64.69 | 47.68 | 60.24 |
| dual-bert-hier-trs(bsz=128, epoch=5, bert-post, bert-trs)| 30.02 | 48.62 | 82.47 | 65.25 | 48.58 | 60.79 |
| dual-bert-hier-trs(bsz=128, epoch=10, bert-post, bert-trs)| 30.78 | 47.87 | 81.58 | 65.21 | 48.28 | 60.98 |


| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| dual-bert-hier(bsz=48, epoch=10) | 28.54  | 48.04  | 83.11  | 63.91  | 45.43  | 59.67 |
| dual-bert-hier(bsz=48, epoch=10, bert-post) | 29.77 | 50.41 | 82.2 | 65.29 | 47.38 | 61.02 |
| dual-bert-hier(bsz=128, epoch=10, bert-post)| 29.63 | 49.03 | 84.31 | 64.83 | 47.23 | 60.95 |
| dual-bert-hier-trs(bsz=128, epoch=10, bert-post, trs+jump)| 33.35 | 50.87 | 83.85 | 67.76 | 52.32 | 63.13 |
| dual-bert(max-len=256, bsz=16, epoch=10, shuffle-ddp) | 28.59 | 47.37  | 81.81 | 63.49 | 45.88 | 59.56  |
| dual-bert(max-len=256, bsz=16, epoch=5, bert-post) | 29.86 | 50.25  | 84.85 | 65.4 | 47.68 | 61.55  |
| dual-bert(max-len=256, bsz=32, epoch=10, shuffle-ddp) | 30.09 | 48.05  | 83.67 | 64.42 | 46.93 | 60.68  |
| dual-bert-adv(max-len=256, bsz=16, epoch=10, shuffle-ddp) | 28.94 | 48.02 | 82.39 | 63.74 | 46.03 | 59.83 |
| dual-bert-one2many(bsz=16, epoch=10, shuffle-ddp) |  | | | |

| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| Bi-Encoder(max-len=512,bsz=16) | 28.1 | 47.85  | 83.05 | 64.45 | 46.63 | 59.36  |
| Bi-Encoder(bsz=16) | 28.16 | 48.5  | 80.87 | 63.64 | 46.18 | 59.38  |
| Bi-Encoder-one2many(bsz=16, bert-post, single-head) | 26.69 | 50.34  | 82.53 | 63.6 | 44.08 | 59.24  |
| Bi-Encoder-one2many-no-addition-loss(bsz=16, bert-post, single-head) | 28.99 | 50.07  | 84.28 | 64.81 | 46.93 | 60.96  |
| Bi-Encoder(bsz=16, bert-post) | 28.9 | 49.51  | 84.7 | 64.62 | 46.63 | 60.98  |
| Bi-Encoder(chinese-bert-wwm,bsz=16) | 28.14 | 46.89  | 81.92 | 63.19 | 45.43 | 59.11  |
| Bi-Encoder(bsz=60) | 30.24 | 50.32 | 83.09 | 65.33 | 47.98 | 61.38  |
| Bi-Encoder-one2many-pseudo(bsz=16,pre-extract=50) | 23.89 | 42.43 | 79.49 | 57.98 | 37.63 | 55.13 |
| Bi-Encoder-one2many(bsz=16,pre-extract=50) | 26.39 | 47.1 | 81.56 | 61.77 | 42.48 | 58.19 |
| Bi-Encoder-one2many-no-additional-loss(bsz=16,pre-extract=50) | 28.89 | 48.39 | 81.13 | 63.4 | 45.28 | 59.63 |
| Bi-Encoder-one2many-concat(bsz=16,pre-extract=200) | 27.2 | 46.04  | 80.46 | 61.83 | 43.63 | 58.33|
| Bi-Encoder-one2many-ivfpq(bsz=16,head=5,max) | 22.53 | 39.03  | 77.26 | 55.92 | 34.93 | 52.71  |
| Bi-Encoder-one2many-lsh(bsz=16,head=5,max) | 28.32 | 46.64  | 81.32 | 62.83 | 44.98 | 58.74  |
| BERT-FT        | 25.86 | 44.63 | 83.43 | 61.55 | 42.58 | 57.59 |
| BERT-FT(bert-post) | 27.51 | 46.67 | 84.39 | 62.92 | 43.93 | 59.46 |
| BERT-FT(bert-post) | | | | | | |
| BERT-FT+MLM+NSP|       |       |       |       |       |       |
| BERT-FT+MLM    |       |       |       |       |       |       |
| BERT-FT+NSP    |       |       |       |       |       |       |

### 3. Ubuntu V1 Dataset

# recall performance
 Because of the very large test set, we use the LSH other than the Flat
| Originali (545868)       | Top-20 | Top-100 | Time |
| -------------- | ----- | ----- | ------ |
| dual-bert-LSH | 0.1374 | 0.2565 | 8.59  |
| dual-bert-fusion-LSH | 0.7934 | 0.8147 | 7.89  |
| ES(q-q) | 0.0101 | 0.0202 | 22.13 |
| ES(q-r) | 0.0014 | 0.0083 | 9.79 |

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 0.884 | 0.946 | 0.990 | 0.975  |
| HCL            | 0.867 | 0.940 | 0.992 | 0.977  |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 0.977  |
| dual-bert(bsz=16, epoch=5, bert-post) | 84.69 | 92.66 | 98.51 | - |
| dual-bert(bsz=64, epoch=5, bert-fp) | 87.64 | 94.22 | 98.57  | - |
| dual-gru(bsz=64, epoch=5) | 72.51 | 85.22 | 96.41 | - |
| dual-bert-hier(bsz=32, epoch=5, bert-post) | 67.71 | 81.33 | 95.36 | - |
| dual-bert-hier(bsz=32, epoch=5, bert-dual-post) | | | | |
| dual-bert-hier(bsz=64, epoch=10) | 79.42 | 89.85 | 97.63 | - |
| dual-bert-hier(bsz=128, epoch=10, bert-post) | 83.14 | 92.04 | 98.32 | - |
| BERT-FT        | | | | |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |

### 5. Writer Dataset

<!-- Inference -->
* Recall performance

| Models                              | Top-20 | Top-100 | Time Cost   |
| ----------------------------------- | ------ | ------- | ----------- |
| dual-bert-gray-writer-LSH           | 0.126  | 0.193   | 529.10      |
| dual-bert-fusion-gray-writer-LSH    |        |         |             |
| hash-bert-gray-writer-BHash512      |        |         |             |

* Rerank performance

| Models                              | R10@1 | R10@2 | R10@5 | MRR   |
| ----------------------------------- | ----- | ----- | ----- | ----- |
| bert-base-chinese (dual-bert-gray-writer) | 66.12 | 79.9 | 94.37 | 77.88 |
| bert-base-chinese (hash-bert-gray-writer) | 54.16 | 75.72 | 94.78 | 71.13 |
| bert-base-chinese (dual-bert|g=2)   | 56.32 | 73.06 | 92.67 | 71.22 |
| bert-base-chinese (dual-bert|g=5)   |  |  |  |  |
| bert-base-chinese (dual-bert|g=10)  |  |  |  |  |
| bert-base-chinese (bert-ft)         |  |  |  |  |
| bert-base-chinese (bert-ft|g=2)     |  |  |  |  |
| hfl-roberta-chinese (dual-bert)     |  |  |  |  |
| hfl-roberta-chinese (dual-bert|g=2) |  |  |  |  |
| hfl-roberta-chinese (dual-bert|g=5) |  |  |  |  |
| hfl-roberta-chinese (dual-bert|g=10)|  |  |  |  |
| hfl-roberta-chinese (bert-ft)       |  |  |  |  |
| hfl-roberta-chinese (bert-ft|g=2)   |  |  |  |  |
| pijili-bert-base (dual-bert)        |  |  |  |  |
| pijili-bert-base (bert-ft)          |  |  |  |  |

### LCCC Dataset

* recall performance

| Models(CPU/394740/full-4769532)    | Top-20 | Top-100 | Time   |
| ---------- | ----- | ----- | --- |
| dual-bert-flat  | 0.105 | 0.195 | 112.39 | 
| dual-bert-fusion-flat  | 0.571 | 0.624  | 110.5 |
| dual-bert-full-LSH  | 0.038 | - | 68.76 | 
| dual-bert-full-fusion-LSH  | 0.417 | 0.475 | 68.92 |
| ES-full(q-q) |  |  |  |
| ES-full(q-r) |  |  |  |
| ES(q-q) | 0.979 | 0.995 | 16.96 |
| ES(q-r) | 0.051 | 0.099 | 9.4 |

* rerank performance

| Models                              | R10@1 | R10@2 | R10@5 | MRR   |
| ----------------------------------- | ----- | ----- | ----- | ----- |
| dual-bert | 40.5 | 75.0 | 92.8 | 63.88 |
| dual-bert-fusion | 40.7 | 73.9 | 92.5 | 63.77 |
