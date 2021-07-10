# The experiments of the Unparallel settings of Dialog Response Selection

## 1. Traditional Comparison Protocol

### 1.1 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| dual-bert          | 33.13 | 53.99 | 86.0  | 68.41 | 51.27 | 64.28  |
| bert-ft            | 31.1  | 54.52 | 86.36 | 67.72 | 49.93 | 63.76  |

### 1.2 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| HCL            | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 97.7   |
| dual-bert      | 88.57 | 95.06 | 99.09 | - |
| bert-ft        | 90.16 | 95.82 | 99.25 | - |

### 1.3 Restoration-200l Dataset

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
| Methods      | Average Human Evaluation | 
| ------------ | ------------------------ |
| BM25+BERT-FP |                          |
| dual-bert+   |                          |

**The kappa among annotators**: 

## 3. Unparallel Comparison Protocol

Use the extra data (out of domain) monolingual samples to enrich the dual-bert index, which cannot be used by the BM25 q-q matching methods.
<!-- 
test set is not used in the faiss index; put the context utterances in the index(faiss and ES q-q matching index) 
EXT means the extra data is used
-->
| Methods          | Average Human Evaluation | 
| ---------------- | ------------------------ |
| BM25+BERT-FP     |                          |
| EXT-dual-bert+   |                          |
