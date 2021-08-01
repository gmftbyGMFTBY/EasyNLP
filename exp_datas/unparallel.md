# The experiments of the Unparallel settings of Dialog Response Selection

## 1. Traditional Comparison Protocol

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| dual-bert          | 92.5  | 97.0  | 99.4  | 95.49 |
| dual-bert-full     | **95.1**  | **98.0**  | **99.6**  | **97.08** |

### 1.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| dual-bert          | 33.13 | 53.99 | 86.0  | 68.41 | 51.27 | 64.28  |           |
| dual-bert-full     | **34.17** | 53.95 | 85.89  | **69.06** | **53.37** | **64.77**  |      |
| bert-ft            | 31.1  | **54.52** | 86.36 | 67.72 | 49.93 | 63.76  |           |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| dual-bert      | 88.57 | 95.06 | 99.09 | - |
| dual-bert-full | 90.36 | 95.79 | 99.18 | - |
| bert-ft        | 90.16 | 95.82 | 99.25 | - |

### 1.4 Restoration-200k Dataset

* ES test set

<!-- + means the post-train has been used;
bert-fp parameters: lr=3e-5; grad_clip=5.0; see0; batch_size=96; max_len=256, min_mask_num=2;
max_mask_num=20; masked_lm_prob=0.15; min_context_length=2; min_token_length=20; epoch=25; warmup_ratio=0.01
pseudo means the utterances in the train.txt is used for psuedo
pseudo(350w) means the extra douban corpus is used for pseudo
ISN means the inner session negative (hard negative)
-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| bert-ft            | 39.22 | 56.6  | 84.54 | 57.63 |
| BERT-FP(bert-ft+)  | 45.77 | 62.19 | 87.3  | 62.68 |
| BERT-FP(bert-ft+)-full-pesudo+  | 44.95 | 63.28 | 88.22  | 62.42 |
| dual-bert          | 45.08 | 61.74 | 87.38 | 62.17 |
| dual-bert+         | 46.85 | 63.3  | 87.36 | 63.48 |
| dual-bert-bow-full+    | 46.45 | 62.36 | 87.09 | 62.97 |
| dual-bert-full+    | 49.82 | 65.62 | 89.05 | 65.73 |
| dual-bert-full+ISN    | 50.72 | 67.75 | 89.66 | 66.79 |
| dual-bert-full-mixup+    | 49.76 | 65.6 | 88.36 | 65.55 |
| dual-bert-pesudo+  | 49.37 | 65.99 | 88.54 | 65.5  |
| dual-bert-full-distributed_gather+    | 49.69 | 65.81 | 89.24 | 65.72 |
| dual-bert-semi-full+    | 50.18 | 66.36 | 88.95 | 65.99 |
| dual-bert-full-unparallel(350w)+    | 50.1 | 66.63 | 88.99 | 66.0 |
| dual-bert-full-fake+    | 50.02 | 66.32 | 88.6 | 65.98 |
| dual-bert-full-pesudo+  | 50.57 | 65.99 | 88.99 | 66.14  |
| dual-bert-full-pesudo(350w)+  | 49.22 | 66.38 | 88.24 | 65.39  |
| dual-bert-one2many+| 47.88 | 63.7  | 88.19 | 64.14 |
| dual-bert-one2many-topk+| 47.1 | 63.85  | 87.52 | 63.65 |
| dual-bert-one2many-full-pseudo+| 51.23 | 67.95  | 89.54 | 66.97 |

**Conclusion**:
    1. full dataset is useful
    2. mixup data augmention useless
    3. bag of the word loss useless
    4. one2many is usefull
    5. full+one2many improve performance further
    6. extended pseudo label useless

* ES test set with human label

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| dual-bert+full     | 56.76 | 73.92 | 93.98 | 76.06 | 62.83 | 74.87  | 24070.67  |
| dual-bert+full-extra-neg     | 57.79 | 75.13 | 93.59 | 76.62 | 63.94 | 75.39  | 24070.67  |
| dual-bert+full(epoc=10)     | 56.41 | 73.64 | 93.73 | 75.64 | 62.32 | 74.52  | 24070.67  |
| dual-bert+full+pseudo     | 56.46 | 75.69 | 93.39 | 76.23 | 62.73 | 74.82  | 24070.67  |
| dual-bert     | 53.96 | 72.29 | 92.4 | 74.06 | 60.00 | 72.76  | 24070.67  |
| dual-bert+grading(0.8)-full     | 57.18 | 74.24 | 93.83 | 76.22 | 62.83 | 75.07  | 24070.67  |
| dual-bert+full-ishn-temp(0.07)     | 56.22 | 76.34 | 93.33 | 76.07 | 61.92 | 75.01  | 23549.8  |
| dual-bert+full-ishn-temp(.07)-epoch(10)-warmup(0.05)     | 54.71 | 75.81 | 94.02 | 75.37 | 60.61 | 74.2  | 23549.8  |
| dual-bert+full-grading     | 55.09 | 75.1 | 94.5 | 75.2 | 60.71 | 74.15  | 23549.8  |
| dual-bert+full-temp(0.07)     | 55.65 | 74.37 | 94.81 | 75.52 | 61.62 | 74.42  | 24070.67  |
| dual-bert+full+mixup      | 56.29 | 75.52 | 93.72 | 76.09 | 62.32 | 74.9  | 24006.35  |
| dual-bert+full+simsce     | 56.24 | 75.64 | 93.4 | 75.87 | 62.12 | 74.84  | 24070.67  |
| dual-bert+full+one2many     | 56.35 | 73.05 | 93.84 | 75.74 | 62.63 | 74.12  | 68816.42  |
| dual-bert+full+ISN | 55.56 | 74.36 | 93.91 | 75.28 | 61.11 | 74.24  | 19678.29  |

## 2. Full-rank Comparison Protocol

The restoration-200k dataset is used for this full-rank comparison protocol.
The corpus is the responses in the train set.

<!-- 
test set is not used in the faiss index; 
put the context utterances in the index(faiss and ES q-q matching index);
INBS is in-batch negative sampling
all means full+ISN
-->
| Methods                     | 1 | 2 | 3 | 4 | 5 | Average Human Evaluation | Average Time Cost | 
| --------------------------- | - | - | - | - | - | ------------------------ | ----------------- |
| BM25(q-q)                   |   |   |   |   |   |                          |                   |
| BM25(q-q, topk=100)+BERT-FP |   |   |   |   |   |                          |                   |
| BERT-FP(full-rank)          |   |   |   |   |   |                          |                   |
| dual-bert(full-rank)        |   |   |   |   |   |                          |                   |
| dual-bert+all(full-rank)    |   |   |   |   |   |                          |                   |
| dual-bert+one2many(full-rank) |   |   |   |   |   |                          |                   |
| dual-bert+all+one2many(full-rank) |   |   |   |   |   |                          |                   |
| dual-bert(topk=100)+BERT-FP |   |   |   |   |   |                          |                   |

**The kappa among annotators**: 

## 3. Unparallel Comparison Protocol

Use the extra data (out of domain) monolingual samples to enrich the dual-bert index, which cannot be used by the BM25 q-q matching methods.
<!-- 
test set is not used in the faiss index; put the context utterances in the index(faiss and ES q-q matching index) 
EXT means the extra data is used
BERT-FP=bert-ft+
-->
| Methods                     | 1 | 2 | 3 | 4 | 5 | Average Human Evaluation | Average Time Cost | 
| --------------------------- | - | - | - | - | - | ------------------------ | ----------------- |
| dual-bert(topk=100, in-dataset-extend)+BERT-FP    |   |   |   |   |   |                          |                   |
| dual-bert(topk=100, in-dataset+out-dataset-extend)+BERT-FP    |   |   |   |   |   |                          |                   |

**The kappa among annotators**: 

## 4. Hyper-parameter recall top-k

Choose the propoer top-k value for the full-rank setting and extended full-rank setting epxeriments
<!-- 
AHE means the average human evaluation
ATC means the average time cost
-->
| Models                  | 1 | 2 | 3 | 4 | 5 | AHE | ATC |
| ----------------------- | - | - | - | - | - | --- | --- |
| BM25(topk=10)+BERT-FP   |   |   |   |   |   |     |     | 
| BM25(topk=100)+BERT-FP  |   |   |   |   |   |     |     | 
| BM25(topk=500)+BERT-FP  |   |   |   |   |   |     |     | 
| BM25(topk=1000)+BERT-FP |   |   |   |   |   |     |     | 

## 5. Appendix

### 5.1 Human Evaluation Standard

| Label |  Meaning |
| ----- | -------- |
| 1     | 回复与用户输入完全没有关联，不属于同一个话题，答非所问，完全跑题（若回复为空，同样视为此类型） | 
| 2     | 回复质量介于1-3之间，难以确定 |
| 3     | 回复内容与用户问题关联度很小，或者回复内容存在以下一种或者多种情况：（1）回复与问题重复或者高度相似；（2）前后内容有一定关联性但是较低；（3）回复没有任何和用户提问相关的内容，信息量少的万能回复。|
| 4     | 回复质量基于3-5之间，难以确定 |
| 5     | 直接回复了问题，前后衔接流畅  |

### 5.2 Time Cost for different settings

<!-- for 1000 test samples -->
| Methods   | Total Time Cost(ms) |
| --------- | --------------------- |
| Pure BM25 | 19503.566 |
| BM25(top=100)+BERT-FP | 168504.7786 |
| dual-bert+all(full-rank) | 211419.1337 |


## 6. How to reproduce our results

### 1.

### 2. generate the samples for human annotations

#### 2.1 pure BM25 (q-q matching)



#### 2.2 BM25 (q-q matching, topk=100) + BERT-FP
