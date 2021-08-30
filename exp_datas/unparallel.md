# The experiments of the Unparallel settings of Dialog Response Selection

## 1. Traditional Comparison Protocol

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| dual-bert          | 92.5  | 97.0  | 99.4  | 95.49 |
| dual-bert-full(5, linear scheduler)  | 94.2  | 98.1  | 99.6  | 97.59 |
| dual-bert-hn(full=5, epoch=10, linear scheduler)  | 95.6  | 98.2  | 99.7  | 97.38 |
| dual-bert-hn(5, cosine scheduler)  | 93.8  | 97.7  | 99.6  | 96.53 |

### 1.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| dual-bert          | 33.13 | 53.99 | 86.0  | 68.41 | 51.27 | 64.28  |           |
| dual-bert-full     | **34.17** | 53.95 | 85.89  | **69.06** | **53.37** | **64.77**  |      |
| dual-bert-full(5)     | 33.38 | 52.43 | 87.73  | 68.44 | 52.32 | 64.33  |      |
| dual-bert-full(5,warmup=0.05)     | 32.68 | 52.93 | 86.96  | 68.03 | 51.27 | 64.1  |      |
| dual-bert-hn(bert-mask-da-dmr, full=5)     | 33.42 | 52.83 | 86.93  | 68.31 | 51.72 | 64.44  |      |
| dual-bert-hn(bert-mask-da-dmr, full=5, warmup=0.)     | 33.89 | 52.9 | 84.83  | 68.88 | 53.07 | 64.52  |      |
| dual-bert-hn(bert-mask-da-dmr, full=5, warmup=0., gray=5)     | 32.21 | 53.26 | 85.44  | 67.41 | 49.78 | 63.69  |      |
| dual-bert-hn-pos(bert-mask-da-dmr, full=5)     | 32.73 | 52.71 | 86.21  | 67.85 | 50.97 | 63.97  |      |
| dual-bert-hn-pos(bert-mask-da-dmr, full=5, warmup=0)     | 32.99 | 51.67 | 86.27  | 67.69 | 51.12 | 63.75  |      |
| dual-bert-hn-pos(bert-mask-da-dmr, full=5, warmup=0, gray=5)     | 32.13 | 52.68 | 85.13  | 67.15 | 49.92 | 63.22  |      |
| dual-bert-full(6)  | **32.93** | 53.95 | 84.93  | **67.73** | **51.27** | **64.49**  |      |
| bert-ft            | 31.1  | **54.52** | 86.36 | 67.72 | 49.93 | 63.76  |           |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| dual-bert      | 88.57 | 95.06 | 99.09 | - |
| dual-bert-full | 90.36 | 95.82 | 99.18 | - |
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
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18   | 14800.94  |
| SA-BERT            | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18   | 14800.94  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| dual-bert+full(cosine scheduler)     | 56.54 | 75.12 | 94.18 | 76.04 | 62.53 | 74.78  | 24070.67  |
| dual-bert+full-warmup(0.05)     | 56.54 | 75.12 | 94.18 | 76.04 | 62.53 | 74.78  | 24070.67  |
| dual-bert-pos+full     | 57.23 | 74.87 | 94.54 | 76.44 | 63.33 | 75.16  | 24070.67  |
| dual-bert+full(fp-mono)     | 57.34 | 74.85 | 93.97 | 76.46 | 63.43 | 75.25  | 24070.67  |
| dual-bert+full(fp-mono-35)     | 57.11 | 75.15 | 93.9 | 76.29 | 63.13 | 75.15  | 24070.67  |
| dual-bert+compfull     | 57.1 | 74.57 | 94.35 | 76.37 | 63.23 | 75.14  | 24070.67  |
| dual-bert-cosine+full     | 55.85 | 73.92 | 94.27 | 75.34 | 61.62 | 74.3  | 24070.67  |
| dual-bert+full-extra-neg     | 57.79 | 75.13 | 93.59 | 76.62 | 63.94 | 75.39  | 24070.67  |
| dual-bert+full(epoc=10)     | 56.41 | 73.64 | 93.73 | 75.64 | 62.32 | 74.52  | 24070.67  |
| dual-bert+full+pseudo     | 56.46 | 75.69 | 93.39 | 76.23 | 62.73 | 74.82  | 24070.67  |
| dual-bert     | 53.96 | 72.29 | 92.4 | 74.06 | 60.00 | 72.76  | 24070.67  |
| dual-bert-one2many     | 52.95 | 71.58 | 93.27 | 73.29 | 58.69 | 72.11  | 24070.67  |
| dual-bert+grading(0.8)-full     | 57.18 | 74.24 | 93.83 | 76.22 | 62.83 | 75.07  | 24070.67  |
| dual-bert+full-ishn-temp(0.07)     | 56.22 | 76.34 | 93.33 | 76.07 | 61.92 | 75.01  | 23549.8  |
| dual-bert+full-ishn-temp(.07)-epoch(10)-warmup(0.05)     | 54.71 | 75.81 | 94.02 | 75.37 | 60.61 | 74.2  | 23549.8  |
| dual-bert+full-grading     | 55.09 | 75.1 | 94.5 | 75.2 | 60.71 | 74.15  | 23549.8  |
| dual-bert+full-temp(0.07)     | 55.65 | 74.37 | 94.81 | 75.52 | 61.62 | 74.42  | 24070.67  |
| dual-bert+full+mixup      | 56.29 | 75.52 | 93.72 | 76.09 | 62.32 | 74.9  | 24006.35  |
| dual-bert+full+simsce     | 56.24 | 75.64 | 93.4 | 75.87 | 62.12 | 74.84  | 24070.67  |
| dual-bert+full+one2many     | 56.19 | 74.8 | 93.75 | 76.01 | 62.42 | 74.73  | 68816.42  |
| dual-bert+full+ISN | 55.56 | 74.36 | 93.91 | 75.28 | 61.11 | 74.24  | 19678.29  |
| dual-bert-comp-hn(warmup=0.05) | 56.65 | 75.12 | 92.52 | 75.81 | 62.63 | 74.83  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05) | 56.16 | 74.57 | 93.43 | 75.59 | 62.02 | 74.61  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono) | 57.31 | 76.79 | 94.43 | 76.83 | 63.03 | 75.81  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, original-bert-mask) | 56.93 | 76.2 | 93.57 | 76.38 | 62.93 | 75.29  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, original-bert-mask, w_delta=2.0) | 57.15 | 75.2 | 93.75 | 76.41 | 63.13 | 75.3  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, original-bert-mask, w_delta=3.0) | 56.84 | 75.24 | 94.31 | 76.24 | 62.73 | 75.1  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, bert-mask[bert-fp-mono], w_delta=3.0) | 57.89 | 75.71 | 94.55 | 76.92 | 63.94 | 75.79  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, bert-mask-dmr[bert-fp-mono], w_delta=3.0) | 56.54 | 75.9 | 93.48 | 76.21 | 62.42 | 75.05  | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, original-bert-mask, w_delta=4.0) | 57.38 | 75.51 | 93.8 | 76.49 | 63.33 | 75.44 | 19678.29  |
| dual-bert-hn-pos(warmup=0.05, epoch=5, gray_cand_num=2, bert-fp-mono, new-pos, original-bert-mask, w_delta=5.0) | 57.89 | 75.41 | 93.75 | 76.84 | 63.94 | 75.68 | 19678.29  |
| dual-bert-hn(warmup=0.05, epoch=5, gray_cand_num=2) | 57.48 | 74.92 | 93.41 | 76.47 | 63.43 | 75.39  | 19678.29  |
| dual-bert-hn(warmup=0.05, epoch=5, bert-mask-da[bert-fp], gray_cand_num=2) | 56.86 | 76.36 | 93.63 | 76.5 | 63.03 | 75.24  | 19678.29  |
| dual-bert-hn(warmup=0.05, epoch=5, bert-mask-da-dmr[bert-fp], gray_cand_num=2) | 57.35 | 76.87 | 93.85 | 76.91 | 63.43 | 75.62  | 19678.29  |
| dual-bert-hn(warmup=0., epoch=2, bert-mask-da-dmr[bert-fp], gray_cand_num=2, no-hard-negative-of-other-samples) | 58.33 | 76.73 | 93.79 | 77.2 | 64.34 | 76.06  | 19678.29  |

## 2. Full-rank Comparison Protocol

The restoration-200k dataset is used for this full-rank comparison protocol.
The corpus is the responses in the train set.

| Methods | R@1000 | R@500 | R@100 | R@50 | MRR |
| ------- | ------ | ----- | ----- | ---- | --- |
| dual-bert | 0.4908       | 0.4098      | 0.2614     | 0.2157     | 0.0692   |
| dual-bert-hn-pos-mono | 0.4935       | 0.414      | 0.2629     | 0.2087     | 0.0701   |
| dual-bert-all-mono | 0.4929       | 0.4056      | 0.2572     | 0.2031     | 0.075   |
| dual-bert-all | 0.5038       |  0.4298     | 0.2624     | 0.2076     | 0.0684   |
| dual-bert-all-extout | 0.521       |  0.443     | 0.2778     | 0.2144     | 0.0705   |
| dual-bert-all-mono-out-dataset(35) | 0.1332       |  0.0998     | 0.0535     | 0.042     | 0.0118   |
| dual-bert-all-out-dataset | 0.2209       |  0.1623     | 0.0899     | 0.0675     | 0.0172   |
| dual-bert-pos-all-out-dataset | 0.2264       |  0.1704     | 0.0895     | 0.0712     | 0.0214   |
| dual-bert-all-bm25-neg-out-dataset | 0.2058       |  0.1543     | 0.0768     | 0.055     | 0.0129   |
| dual-bert-out-dataset | 0.1645       |  0.1151     | 0.0578     | 0.0395     | 0.0118   |
| dual-bert-bert-mask-aug-out-dataset | 0.2531       |  0.1932     | 0.0937     | 0.0714     | 0.0201   |
| dual-bert-proj-bert-mask-aug-out-dataset | 0.0798      |  0.0552     | 0.0193     | 0.015     | 0.002   |


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

### 5.3 Fine-grained test results

|   Methods   | SBM lt | Weight Kendall Tau lt | SBM brandenwang | Weight Kendall Tau brandenwang | SBM lt2 | WKT lt |
| ----------- | --------- | ------------------ | ------- | ------ | ---- | ----- |
| bert-ft   | 0.6437    | 0.1447               | 0.6032  | 0.0713 | 0.534  | -0.0215  |
| dual-bert   | 0.6892    | 0.2262               | 0.6727  | 0.1761 | 0.435  | -0.1418  |
| dual-bert-full | 0.6811 | 0.1965             | 0.6788  | 0.2231 | 0.4526  | -0.0878  |
| dual-bert-full+bert-fp-mono(35) | 0.6916 | 0.2362      | 0.6802  | 0.1645 |  |   |
| dual-bert-pos-mono | 0.6949  |             | 0.6665  |  | 0.4519  |  |
| dual-bert-hn | 0.695  | 0.2293             | 0.6807  | 0.1943  |   |   |
| dual-bert-hn-warmup(bert-mask-da: 0.5 mask ratio, 5 aug_t, 20 maxium mask num) | 0.6865  | 0.1825   | 0.6666  | 0.2144  | 0.4591 | -0.1259 |
| dual-bert-hn-pos(0.5)  | 0.6983 |    | 0.6914  | | 0.4744 |  |
| dual-bert-hn-pos(0.5,修改了部分bert-mask的代码)  | 0.6939 |  | 0.6669  | | | |
| dual-bert-hn-pos  | 0.6883 |   | 0.6911  | | 0.4618 | |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, original-bert-mask, gray_cand_num=2, bert-fp-mono, w_delta=1.0)  | 0.6912 | 0.2279          | 0.6861  | 0.2092 | 0.4702 | -0.0949 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, original-bert-mask, gray_cand_num=2, bert-fp-mono, w_delta=2.0)  | 0.6926 | 0.2369          | 0.6899  | 0.2359 | 0.4749 | -0.1034 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, original-bert-mask, gray_cand_num=2, bert-fp-mono, w_delta=3.0)  | 0.7037 | 0.2299          | 0.7096  | 0.2099 | 0.4749 | -0.1365 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, bert-mask[bert-fp-mono], gray_cand_num=2, bert-fp-mono, w_delta=3.0)  | 0.6689 | 0.1643          | 0.6682  | 0.2585 | 0.5107 | -0.1242 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, bert-mask-dmr[bert-fp-mono], gray_cand_num=2, bert-fp-mono, w_delta=3.0)  | 0.6945 | 0.2081          | 0.6733  | 0.1905 | 0.4722 | -0.1506 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, original-bert-mask, gray_cand_num=2, bert-fp-mono, w_delta=4.0)  | 0.689 | 0.2193          | 0.6901  | 0.2366 | 0.4752 | -0.0868 |
| dual-bert-hn-pos(warmup=0.05, epoch=5, new-pos, original-bert-mask, gray_cand_num=2, bert-fp-mono, w_delta=5.0)  | 0.6885 | 0.2139          | 0.6844  | 0.2139 | 0.4745 | -0.0727 |
| dual-bert-comp  | 0.6869 |  | 0.665  | | | |
| dual-bert-comp-hn  | 0.6891 |   | 0.6802  | | | |
| dual-bert-proj  | 0.6923 | 0.2044          | 0.6807  | 0.1918 | 4509 | -0.0561 |
| dual-bert-hn  | 0.7061 | 0.2132          | 0.6832  | 0.2042 | 0.4812 | -0.1266 |
| dual-bert-hn(bert-mask-da-dmr[bert-fp])  | 0.6959 | 0.238          | 0.6845  | 0.2264 | 0.4671 | -0.1044 |



## 6. How to reproduce our results

### 1.

### 2. generate the samples for human annotations

#### 2.1 pure BM25 (q-q matching)



#### 2.2 BM25 (q-q matching, topk=100) + BERT-FP
