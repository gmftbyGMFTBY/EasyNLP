# The experiments of the Unparallel settings of Dialog Response Selection

## 1. Traditional Comparison Protocol

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| dual-bert          | **92.5**  | **97.0**  | **99.4**  | **95.49** |
| bert-ft            |       |       |       |       |

### 1.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | **51.4**  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| dual-bert          | **33.13** | 53.99 | 86.0  | **68.41** | 51.27 | 64.28  |
| bert-ft            | 31.1  | **54.52** | 86.36 | 67.72 | 49.93 | 63.76  |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| dual-bert      | 88.57 | 95.06 | 99.09 | - |
| bert-ft        | 90.16 | 95.82 | 99.25 | - |

### 1.4 Restoration-200l Dataset

* ES test set

<!-- + means the post-train has been used;
bert-fp parameters: lr=3e-5; grad_clip=5.0; see0; batch_size=96; max_len=256, min_mask_num=2;
max_mask_num=20; masked_lm_prob=0.15; min_context_length=2; min_token_length=20; epoch=25; warmup_ratio=0.01-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| dual-bert          | 45.08 | 61.74 | 87.38 | 62.17 |
| bert-ft            | 39.22 | 56.6  | 84.54 | 57.63 |
| dual-bert+         |  |  |  |  |
| bert-ft+           |  |  |  |  |

* ES test set with human label

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| dual-bert          | | | | |
| bert-ft            | | | | |
| dual-bert+         | | | | |
| bert-ft+           | | | | |

## 2. Full-rank Comparison Protocol

The restoration-200k dataset is used for this full-rank comparison protocol

<!-- 
test set is not used in the faiss index; 
put the context utterances in the index(faiss and ES q-q matching index) 
-->
| Methods          | 1 | 2 | 3 | 4 | 5 | Average Human Evaluation | 
| ---------------- | - | - | - | - | - | ------------------------ |
| BM25+BERT-FP     |   |   |   |   |   |                          |
| dual-bert+   |   |   |   |   |   |                          |

**The kappa among annotators**: 

## 3. Unparallel Comparison Protocol

Use the extra data (out of domain) monolingual samples to enrich the dual-bert index, which cannot be used by the BM25 q-q matching methods.
<!-- 
test set is not used in the faiss index; put the context utterances in the index(faiss and ES q-q matching index) 
EXT means the extra data is used
BERT-FP=bert-ft+
-->
| Methods          | 1 | 2 | 3 | 4 | 5 | Average Human Evaluation | 
| ---------------- | - | - | - | - | - | ------------------------ |
| BM25+BERT-FP     |   |   |   |   |   |                          |
| EXT-dual-bert+   |   |   |   |   |   |                          |

**The kappa among annotators**: 

## 4. Appendix

### 4.1 Human Evaluation Standard

| Label |  Meaning |
| ----- | -------- |
| 1     | 回复与用户输入完全没有关联，不属于同一个话题，答非所问，完全跑题（若回复为空，同样视为此类型） | 
| 2     | 回复质量介于1-3之间，难以确定 |
| 3     | 回复内容与用户问题关联度很小，或者回复内容存在以下一种或者多种情况：（1）回复与问题重复或者高度相似；（2）前后内容有一定关联性但是较低；（3）回复没有任何和用户提问相关的内容，信息量少的万能回复。|
| 4     | 回复质量基于3-5之间，难以确定 |
| 5     | 直接回复了问题，前后衔接流畅  |
