# The experiments of SCM

## 1. Traditional Comparison Protocol

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| DR-BERT            | 96.0  | 98.4  | 99.6  | 92.52 |
| dual-bert-scm(epoch=10,bert-fp-mono,bsz=64,full=9)      |  96.4 | 98.7  | 99.7  | 28.09 |

### 1.2 Douban Dataset

<!-- ablation study of the fine-grained degree -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| DR-BERT(bert-fp-mono,epoch-5,full=5)            | 34.65  | 54.84  | 86.03  | 69.29  | 52.92  | 65.53   |  24.72        |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| bert-ft        | 90.16 | 95.82 | 99.25 | - |
| DR-BERT(full=100,bert-fp-mono,epoch=5)      | 91.25 | 96.08 | 99.3 | - |

### 1.4 Restoration-200k Dataset


<!-- validation during training -->
<!-- conclusio: (1) more fusion encoder layer is not useful, leads to lower validation performance; (2) hard negative is necessary; (3) more heads in MHA seems useful; (4) context-aware before comparison is useful; (5) bigger transformer decoder capacity is not useful (SCMHN2); (5) innert context negative is bad; -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  | 14800.94  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  | 15135.32  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| DR-BERT(bsz=80)    | 58.08 | 75.86 | 94.42 | 77.81 | 64.34 | 75.82   | 24.23  |
| DR-BERT+SCM(bm25-hn,bsz=80)       | 58.95 | 76.81 | 93.85 | 77.5 | 64.65 | 76.33  | 28.15  |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=80,easy-hn)       | 55.56 | 74.84 | 93.74 | 75.56 | 61.62 | 74.22 | 25.29 |
| DR-BERT+SCM(nhead=8,nlayer=2,bsz=80)       | 57.27 | 74.65 | 94.47 | 76.5 | 63.33 | 75.21 | 26.55 |
| DR-BERT+SCM(nhead=8,nlayer=4,gray=2,bsz=80)       | 58.95 | 76.81 | 93.85 | 77.50 | 64.65 | 76.33 | 26.45 |
| DR-BERT+SCM(nhead=8,nlayer=2,gray=2,bsz=80)       | 57.52 | 77.27 | 94.33 | 77.11 | 63.23 | 75.98 | 26.45 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)       | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=80,context-aware-before-comparison)       | 58.68 | 77.12 | 94.4  | 77.73 | 64.75 | 76.56 | 25.53 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=80,context-aware-before-comparison,bm25-q-r-hn)       | 55.88 | 74.07 | 93.31  | 75.24 | 61.52 | 74.2 | 25.52 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=4,bsz=80,context-aware-before-comparison)       | 57.38 | 76.73 | 93.43 | 76.68 | 63.13 | 75.46 | 25.14 |
| DR-BERT+SCM(nhead=8,nlayer=1,bm25+ctx-hn,gray=2,bsz=80,context-aware-before-comparison)       | 55.86 | 74.82 | 94.25 | 75.84 |  62.02| 74.57 | 24.9 |
| DR-BERT+SCM(nhead=12,nlayer=1,gray=2,bsz=80,context-aware-before-comparison)       | 57.13 | 76.18 | 93.66  | 76.67 | 63.23 | 75.35 | 25.13 |
| DR-BERT+SCM(residual,bsz=80)       | 56.06 | 75.37 | 93.47 |76.06  | 62.32 | 74.67  | 28.4  |
| DR-BERT+SCM(nhead=12,bsz=80)      | 55.82 | 75.3 | 94.1 | 75.83 | 61.92 | 74.52 | 28.84 |
| DR-BERT+SCM(nhead=6,bsz=80)       | 56.59 | 76.88 | 93.3 | 76.5 | 62.93 | 75.13 | 28.19 |
| DR-BERT+SCM-MCH(nhead=8,fusion_layer=5,gray=2,bsz=80)       | 58.19 | 75.2 | 93.52 | 76.93 | 64.24 | 75.67 | 29.92 |
| DR-BERT+SCM-MCH(nhead=8,fusion_layer=5,gray=4,bsz=80)       | 58.04 | 75.68 | 92.81 | 76.81 | 63.74 | 75.64 | 30.58 |

<!-- batch size ablation study -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(dynamic_mask[10-128],nhead=8,nlayer=1,gray=2,bsz=128)      | 58.48 | 77.38 | 94.13 | 77.64 | 64.55 | 76.33 | 26.02 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)      | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=112)      | 58.61 | 76.78 | 94.75 | 77.64 | 64.75 | 76.47 | 25.46 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=96)       | 58.36 | 77.01 | 94.28 | 77.52 | 64.34 | 76.32 | 25.21 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=80)       | 58.14 | 77.04 | 93.68 | 77.28 | 64.04 | 76.09 | 26.21 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=64)       | 59.03 | 77.57 | 94.28 | 77.92 | 65.05 | 76.71 | 25.34 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=48)       | 57.89 | 78.49 | 93.82 | 77.4  | 63.64 | 76.29 | 25.21 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=32)       | 58.68 | 76.83 | 93.4  | 77.43 | 64.55 | 76.17 | 24.85 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=16)       | 58.29 | 75.18 | 93.64 | 76.84 | 63.84 | 75.58 | 25.55 |

<!-- head num ablation study -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=2,nlayer=1,gray=2,bsz=80)       | 58.19 | 76.69 | 93.5  | 77.24 | 64.04 | 76.15 | 25.2 |
| DR-BERT+SCM(nhead=4,nlayer=1,gray=2,bsz=80)       | 58.8  | 77.16 | 93.88 | 77.6  | 64.85 | 76.32 | 25.19 |
| DR-BERT+SCM(nhead=8,nlayer=2,gray=1,bsz=80)       | 57.95 | 77.26 | 94.74 | 77.26 | 64.94 | 76.17 | 26.25 |
| DR-BERT+SCM(nhead=12,nlayer=1,gray=2,bsz=80)      | 58.96 | 77.39 | 93.83 | 77.73 | 64.85 | 76.69 | 25.05 |

<!-- nlayer ablation study -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)       | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=2,gray=2,bsz=128)       | 59.39 | 79.12 | 95.02 | 78.71 | 65.76 | 77.39 | 26.27 |
| DR-BERT+SCM(nhead=8,nlayer=3,gray=2,bsz=128)       | 57.62 | 77.67 | 94.43 | 77.24 | 63.64 | 76.23 | 27.54 |
| DR-BERT+SCM(nhead=8,nlayer=4,gray=2,bsz=128)       | 56.71 | 75.93 | 93.31 | 76.07 | 62.22 | 75.0  | 28.61 |
| DR-BERT+SCM(nhead=8,nlayer=5,gray=2,bsz=128)       | 56.99 | 73.29 | 92.48 | 75.51 | 62.42 | 74.39 | 29.83 |
| DR-BERT+SCM(mch,nhead=8,nlayer=1,gray=2,bsz=128)   | 59.28 | 76.41 | 94.3 | 77.93 | 65.45 | 76.74 | 25.6 |
| DR-BERT+SCM(mch,nhead=8,nlayer=2,gray=2,bsz=128)   | 59.03 | 77.55 | 94.23 | 78.0 | 65.15 | 76.78 | 26.03 |
| DR-BERT+SCM(mch,nhead=8,nlayer=3,gray=2,bsz=128)   | 58.46 | 77.64 | 93.86 | 77.63 | 64.55 | 76.48 | 27.53 |
| DR-BERT+SCM(mch,nhead=8,nlayer=4,gray=2,bsz=128)   | 58.83 | 75.91 | 94.0 | 77.37 | 64.65 | 76.28 | 28.73 |
| DR-BERT+SCM(mch,nhead=8,nlayer=5,gray=2,bsz=128)   | 57.92 | 76.07 | 93.81 | 76.98 | 63.84 | 75.86 | 30.32 |

<!-- gray num ablation study -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)       | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=4,bsz=128)       | 58.29 | 75.95 | 93.41 | 77.11 | 64.24 | 75.94 | 25.13 |


<!-- transformer encoder or decoder -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)       | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128,trs-enc)       | 58.21 | 77.72 | 93.99 | 77.56 | 64.24 | 76.42 | 24.62 |

<!-- transformer encoder batch size -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128,trs-enc)       | 58.21 | 77.72 | 93.99 | 77.56 | 64.24 | 76.42 | 24.62 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=256,trs-enc)       |  |  |  |  |  |  |  |


<!-- before comparison ablation study -->
<!-- | DR-BERT+SCM(nhead=8,nlayer=2,gray=2,bsz=128,before_comp)       | 58.83 | 77.62 | 94.13 | 77.81 | 64.95 | 76.82 | 26.4 | -->
<!-- | DR-BERT+SCM(nhead=12,nlayer=2,gray=2,bsz=128,before_comp)      | 59.81 | 75.88 | 94.07 | 77.91 | 65.86 | 76.9 | 26.72 | -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128)       | 59.22 | 77.89 | 94.43 | 78.18 | 65.35 | 76.91 | 25.04 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=128,before_comp)       | 60.4 | 78.27 | 94.33 | 78.71 | 66.46 | 77.71 | 25.73 |
| DR-BERT+SCM(nhead=12,nlayer=1,gray=2,bsz=128,before_comp)      | 58.49 | 77.39 | 93.85 | 77.63 | 64.65 | 76.54 | 25.98 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=64)       | 59.03 | 77.57 | 94.28 | 77.92 | 65.05 | 76.71 | 25.34 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=64,before_comp)       | 58.51 | 78.01 | 94.70 | 77.72 | 64.55 | 76.68 | 25.14 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=32)       | 58.68 | 76.83 | 93.4  | 77.43 | 64.55 | 76.17 | 24.85 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=32,before_comp)       | 59.42 | 77.59 |  94.25 | 78.12 |65.45  | 76.89 | 25.4 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=16)       | 58.29 | 75.18 | 93.64 | 76.84 | 63.84 | 75.58 | 25.55 |
| DR-BERT+SCM(nhead=8,nlayer=1,gray=2,bsz=16,before_comp)       | 58.02 | 76.51 | 93.91 | 77.26 | 64.14 | 75.95 | 24.66 |
