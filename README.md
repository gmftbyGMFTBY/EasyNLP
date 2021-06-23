## State-of-the-art retrieval-based multi-turn response selection baselines

Constrastive Learning for dual-encoder model, leveraging the memory bank to enlarge the number of the negative samples. Besides, the FAISS memory bank can be used to recall better samples, so these better samples can be used as the medium samples (not so good and not so bad).

Constrastive Learning, context is the q, and response is the jey. Ground-truth response is positive k, and other responses are negative k.

NOTE:
- [ ] Constrastive learning on huge dataset, LCCC, and fine-tuning on ecommerce and douban corpus. But need to compared with the dual-bert pre-trained model
- [x] test on Ubuntu v1 corpus
- [x] for Ubuntu v1 corpus, add the special tokens for the BertTokenizer during fine-tuning
- [x] larger batch size for dual-bert-hierarchical model on ecommerce, douban, ubuntu (128)
- [x] speaker embedding in dual-bert-hierarchical (necessary?)
- [x] fully use all the utterances, the sequence length more than 64 will be cut.
- [x] refer to MSN, IoI, ... for turn-aware aggregation
- [x] jump connection is essential
- [x] dual post train for the dual-bert-hierarchical. The bert-post checkpoint may not be appripriate for the dual-encoder architecture. So, the dual-bert-post should be used for dual-bert-hierarchical or dual-bert-hierarchical-trs model, which train the dual-bert model with the bert-post initilized.
- [x] pytorch 1.5.1+cu92 (CUDA 9.2), apex 0.1, NVIDIA driver: 410.78 (CUDA Version: 10.0)
- [ ] test cross-encoder-hierarchical (bert-ft-hierarchical) further
- [ ] new idea: during training, dual encoder architecture doesn't memory the former training samples, which leads to the 较低的样本利用率，这个强化学习是一样的。改进低样本利用率。样本利用率还是负样本的数目？
- [ ] 把检索式对话系统做成QA的选择题，而不是现在的对错题
- [ ] 基于文档的对话系统使用层次化方法，充分考虑所有对话句子以及文档信息，可以看成embedding方法全部从word embedding换成了bert
- [ ] 仔细对比以下dual encoder架构和cross-encoder架构，传统的方法也用dual encoder架构复现以下，主要是证明一下in-batch negative sample方法的有效性。
- [ ] test the difference between 5 and 10
- [ ] 测试蒸馏学习解决层次化的无法建模细粒度信息的问题
- [ ] 测试不同step下的dual-bert-hier和dual-bert的效果（检测训练效率，同bsz e.g. negative samplers）
- [ ] 直接继承 dual-bert-hier 的dataloader写bert-ft-hier
- [ ] dual-bert-hier 之前用 last utterance 作为 key 重新获得细粒度的 history 在做 contextual transformer encoding
- [x] dual-bert-hier-trs-poly and dual-bert-hier-trs-poly2 sucks
- [ ] 换个角度思考，就算dual encoder architecture效果不如cross encoder，但是dual encoder因为独有的有点可以作为粗筛的一个有效方法，那么只要改进在dual encoder architecture是显著的就行，不一定要打败SOTA，只要比dual encoder本身好就行
- [x] the hard negative seems not work as well
- [ ] test the short conversation context (multiple samples) for dual-bert
- [x] Conlusion of the hard negative samples: (1) BERT-FP (utterance in the context); (2) BERT-HCL (semantic recall); (3) WorldNotBinary (Generation, retrieval[BM25], random sample)
- [ ] remove label smoothing and test

## How to Use

Before using the repo, please run the following command to init:

1. Init the repo

```bash
./run.sh init
```

2. create post data from the orignial dataset (optional)

```bash
cd data;
python create_post_data.py
```

3. create data for post training (optional)

```bash
# dataset saved in data/ecommerce/train_post.hdf5
./data/create_post_train_dataset.sh ecommerce
```

4. post train (optional)

```bash
# checkpoint are saved in ckpt/ecommerce/bert-post/*
./post_train/post-train.sh ecommerce <nspmlm/mlm/nsp> <gpu_ids>
```

5. load post train checkpoint and fine-tuning on response selection corpus

    It should be noted that, the newest post train checkpoint can be directly loaded from the [BERT-FP model](https://github.com/hanjanghoon/BERT_FP). Just put it into `ckpt/<dataset_name>/<model_name>/best_nspmlm.pt`.

```bash
./run.sh train-post ecommerce bert-ft <gpu_ids>
```

6. inference on the train dataset, save in FAISS index

```bash
# save the extended train dataset into data/<dataset>/candidates.pt
./run.sh inference ecommerce dual-bert <gpu_ids>

# if you need to save the context and response (qa) into the FAISS index
# run the following command
./run.sh inference_qa ecommerce dual-bert <gpu_ids>
```

7. train the model, test after each epoch

    after training, the log information during the training procedure are saved in the tensorboard file, which can be found under `rest/<dataset_name>/<model_name>/`

```bash
./run.sh train ecommerce dual-bert <gpu_ids>

# writer dataset now support two kinds of models: dual-bert-writer, sa-bert-neg
./run.sh train writer dual-bert-writer 0,1,2,3,4,5,6,7
./run.sh train writer bert-ft-writer 0,1,2,3,4,5,6,7
```
8. test the model

```bash
./run.sh test ecommerce dual-bert <gpu_ids>
```

## Ready models and datasets

### 1. Models
1. BERT-ft
2. BERT-ft-gen
3. Bi-Encoder
4. Poly-Encoder
5. Bi-Encoder+one2many


### 2. Datasets
1. E-Commerce
2. Douban
3. Ubuntu-v1
4. LCCC
5. Writer (智能写作助手数据集)


## Experiment Results
### 1. E-Commerce Dataset

_Note:_
* Bi-Encoder and Poly-Encoder use in-batch negative approach during training procedure, and the number of negative samples is equal to the bsz (default 16) minus 1.
* batch size is 16
* for Bi-Encoder-VAE, max strategy is better than mean
* sequence length is 256
* Compared with Bi-Encoder-CL, Bi-Encoder-CL2 fully leverage the variant response represeantion
* more negative samples, better performance
* google position embedding is better than absolute position embedding
* max strategy is equal to mean strategy
* worse quality (bi-encoder-one2many-bad) of the candidates could bring better performance (test whether the number of the negative samples matter, rather than the quality)
* the number of the samples and the quality of samples matter!!!
* adding the number of the positive samples doesn't improve the performance!!!
* context max length 256, response max length 128


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

* one2many performance is worse than dual-bert on douban corpus, which is very different from the ecommerce corpus. The reason maybe: the dual-bert performance is worse than that on ecommerce corpus, which lead to bad candidate samples for dual-bert-one2many model, e.g., the quality of the candidates matter!
* max-sequence-length and PLMs (hfl/chinese-bert-wwm is slightly worse than bert-base-chinese) is not the core, but max-sequence-length do improve some metrics
* good candidates (pre-extract=20, topk=10) provide  better R10@1 and R10@2, hopeful!
* context max length 256, response max length 128

| Model (CPU/68w)        | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat         | 0.078  | 0.1634  | 255.63              |
| dual-bert-IVF8192,Flat | 0.057  | 0.0795  | 16.91               |
| dual-bert-IVF100,Flat  | 0.0675 | 0.1199  | 29.62               |
| dual-bert-LSH          | 0.1139 | 0.1934  | 20.36               |
| hash-bert-flat         | 0.045  | 0.1109  | 13.43               |
| hash-bert-BHash512     | 0.0435 | 0.1064  | 7.26                |

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


| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP(512)       | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| bert-ft(512)       | 32.27 | 53.87 | 85.64 | 67.86 | 50.52 | 64.24  |
| poly encoder(poly-m=32)  | 31.76 | 50.59 | 85.72 | 66.49 | 49.48 | 62.84  |
| dual-bert(bsz=32, epoch=5, poly-m=32) | 30.55 | 46.93 | 81.16 | 64.45 | 48.43  | 60.34 |
| dual-bert-layer(bsz=32, epoch=5, bert-fp) | 28.95 | 50.41 | 83.2 | 64.99 | 46.93  | 60.7 |
| dual-bert(bsz=32, epoch=5, bert-fp) | 31.42 | 51.6 | 83.46 | 66.41 | 49.48  | 62.22 |
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

### 3. LCCC Dataset

* max sequence length: 256

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | -     | -     | -     | -      |
| Bi-Encoder(bsz=16) | 21.5  | 31.2      | 47.1   | 36.65   |
| Bi-Encoder-one2many(bsz=16,max,pre-extract=50) | 27.0 | 36.7 | 54.0 | 41.85 |
| BERT-FT        | 23.1  | 72.1  | 94.2  | 54.9   |
| BERT-FT+MLM+NSP|       |       |       |        |
| BERT-FT+MLM    |       |       |       |        |
| BERT-FT+NSP    |       |       |       |        |

### 4. Ubuntu V1 Dataset

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 0.884 | 0.946 | 0.990 | 0.975  |
| HCL            | 0.867 | 0.940 | 0.992 | 0.977  |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 0.977  |
| dual-bert(bsz=16, epoch=5, bert-post) | 84.69 | 92.66 | 98.51 | - |
| dual-bert(bsz=32, epoch=5, bert-fp) | 87.44 | 94.12 | 98.5  | - |
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
| Models                              | Top-20 | Top-100 | Time Cost   |
| ----------------------------------- | ------ | ------- | ----------- |
| dual-bert-gray-writer-LSH           |        |         |             |
| dual-bert-gray-writer-IVF100,PQ16   |        |         |             |
| hash-bert-gray-writer-BHash512      |        |         |             |

| Models                              | R10@1 | R10@2 | R10@5 | MRR   |
| ----------------------------------- | ----- | ----- | ----- | ----- |
| bert-base-chinese (dual-bert-gray-writer) | 66.12 | 79.9 | 94.37 | 77.88 |
| bert-base-chinese (hash-bert-gray-writer) | | | | |
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
