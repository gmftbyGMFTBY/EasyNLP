# The experiments of the Unparallel settings of Dialog Response Selection

## 1. Traditional Comparison Protocol

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| poly-encoder       |  92.4  |  96.3 | 99.2 | 95.32 |
| poly-encoder-full(5)| 93.8 | 98.5  | 99.5  | 96.49 |
| dual-bert          | 92.5  | 97.0  | 99.4  | 95.49 |
| dual-bert-full(5, lr=3e-5) | 93.1  | 97.6  | 99.7  | 95.96 |
| dual-bert-full(5, lr=5e-5) | 94.4  | 98.3  | 99.5  | 96.76 |
| dual-bert-full(5, lr=8e-5) | 94.7  | 98.2  | 99.7  | 96.89 |
| dual-bert-hn(full=5, epoch=10, linear scheduler)  | 95.6  | 98.2  | 99.7  | 97.38 |
| dual-bert-hn(epoch=5,gray=2,cosine[temp=0.07],bert-mask-da-dmr[aug_t=10,0.4-0.6])  | 95.4  | 98.1  | 99.4  | 97.2 |
| dual-bert-hn(no-fg,epoch=5,gray=2,cosine[temp=0.07],bert-mask-da-dmr[aug_t=10,0.4-0.6])  | 91.3  | 96.2  | 99.1  | 94.62 |

<!-- ablation study -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| dual-bert          | 92.5  | 97.0  | 99.4  | 95.49 |
| dual-bert-full(full=5,lr=5e-5,bert-fp,epoch=5) | 91.1  | 97.0 | 99.5 | 94.87 |
| dual-bert-full(full=5,lr=5e-5,bert-fp-mono,epoch=5) | 93.6  | 97.7  | 99.6  | 96.25 |
| dual-bert-full(full=5,lr=5e-5,bert-fp,epoch=10) | 94.9  | 98.4  | 99.7 | 97.6 |
| dual-bert-full(full=5,lr=5e-5,bert-fp-mono,epoch=10) | 95.9 | 98.7 | 99.5 | 97.59 |
| dual-bert-full(no-full,lr=5e-5,bert-fp-mono,epoch=10) | 92.5 | 96.9| 99.5 | 95.52 |

<!-- ablation study on fine-grained degree -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| dual-bert-full(no-full,lr=5e-5,bert-fp-mono,epoch=10)| 92.5 | 96.9 | 99.5 | 95.52 |
| dual-bert-full(full=2,lr=5e-5,bert-fp-mono,epoch=10) | 91.5 | 96.9 | 99.6 | 95.01 |
| dual-bert-full(full=3,lr=5e-5,bert-fp-mono,epoch=10) | 95.2 | 98.5 | 99.6 | 95.01 |
| dual-bert-full(full=5,lr=5e-5,bert-fp-mono,epoch=10) | 95.9 | 98.7 | 99.5 | 97.59 |
| dual-bert-full(full=7,lr=5e-5,bert-fp-mono,epoch=10) | 95.6 | 98.9 | 99.8 |  97.5 |
| dual-bert-full(full=9,lr=5e-5,bert-fp-mono,epoch=10) | 96.0 | 98.4 | 99.6 |  97.6 |

### 1.2 Douban Dataset

<!-- ablation study of the fine-grained degree -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| DR-BERT(bert-fp-mono,epoch-5,full=1)            | 31.07  | 49.59  | 83.4  | 65.54  | 48.73  | 61.3   |  25.03        |
| DR-BERT(bert-fp-mono,epoch-5,full=3)            | 32.51  | 53.55  | 87.15 | 68.34  |  51.72 | 64.32  |   24.99     |
| DR-BERT(bert-fp-mono,epoch-5,full=5)            | 34.74  | 53.98  | 86.14  | 69.44  | 53.97  | 65.23   |  25.15        |
| DR-BERT(bert-fp-mono,epoch-5,full=7)            | 33.89 | 53.64  | 86.26 | 68.9  | 52.77  | 65.08  |  24.98      |
| DR-BERT(bert-fp-mono,epoch-5,full=9)            | 33.03 | 54.39  | 87.29 | 68.21  | 51.27  | 64.79  |  29.24      |

<!-- ablation study of the bert-fp-mono -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| DR-BERT(bert-fp-mono,epoch-5,full=5)            | 34.74  | 53.98  | 86.14  | 69.44  | 53.97  | 65.23   |  25.15        |
| DR-BERT(bert-fp-mono,epoch-5,full=5)            | 34.65  | 54.84  | 86.03  | 69.29  | 52.92  | 65.53   |  24.72        |
| DR-BERT(no-full,bert-fp-mono,epoch-5)            | 31.07  | 49.59  | 83.4  | 65.54  | 48.73  | 61.3   |  25.03        |

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |           |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | 64.4   |  64775.34        |
| poly-encoder | 29.99 | 49.35 | 82.19 | 64.97 | 47.53 | 60.84 | 30525.56  |
| poly-encoder-full(5) | 30.26 | 50.27 |  84.87 | 65.95 | 48.43 | 61.73 |  29469.71  |
| dual-bert(bertp-fp-mono)          | 33.72 | 53.67 | 86.56  | 68.57 | 52.32 | 64.56  |           |
| dual-bert-full     | **34.17** | 53.95 | 85.89  | **69.06** | **53.37** | **64.77**  |      |
| dual-bert-hn       | 3305 | 53.05 | 86.59  | 68.14 | 51.12 | 64.23  |      |
| dual-bert-full(5)     | 33.38 | 52.43 | 87.73  | 68.44 | 52.32 | 64.33  |      |
| dual-bert-full(5, lr=8e-5, bert-fp)     | 33.93 | 52.57 | 86.59  | 68.47 | 52.47 | 64.66  |      |
| dual-bert-full(5, lr=1e-4, epoch=5, bert-fp, test_interval=0.001)     | 34.41 | 53.6 | 86.18  | 69.19 | 53.52 | 65.0  |      |
| dual-bert-full(5, lr=1e-4, epoch=10, bert-fp, test_interval=0.01)     | 33.84 | 54.13 | 86.14  | 68.79 | 52.32 | 64.73  |      |
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
| bert-ft        | 90.16 | 95.82 | 99.25 | - |
| poly-encoder | 88.15 | 94.86 | 99.04 | - |
| poly-encoder-full(5) | 90.12 | 95.76 | 99.21 | - |
| dual-bert      | 88.57 | 95.06 | 99.09 | - |
| dual-bert-full | 90.36 | 95.82 | 99.18 | - |
| dual-bert-full(bert-fp-mono) | 90.1 | 95.55 | 99.03 | - |
| DR-BERT(full=5,bert-fp-mono,epoch=5)      | 90.52 | 95.74 | 99.17 | - |
| DR-BERT(full=5,bert-fp-mono,epoch=10)      | 90.52 | 95.86 | 99.2 | - |
| DR-BERT(full=100,bert-fp-mono,epoch=5)      | 91.25 | 96.08 | 99.3 | - |
| DR-BERT(full=100,bert-fp-mono,epoch=10)     | 91.0 | 95.95 | 99.25 | - |
| DR-BERT(no-full,bert-fp-mono,epoch=5)      | 88.86 | 94.99 | 99.11 | - |

### 1.4 Restoration-200k Dataset


<!-- validation during training -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  | 14800.94  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  | 15135.32  |
| SA-BERT+BERT-FP    | 51.46 | 69.43 | 92.82 | 71.99 | 57.07 | 70.72  | 15135.32  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| BERT-FP(full)      | 50.45 | 70.4  | 92.82 | 71.72 | 56.06 | 70.37  | 14.74  |
| BERT-FP-SCM(full)  | 51.88 | 67.69 | 89.7  | 71.82 | 57.55 | 70.2   | 14.74  |
| DR-BERT(full)      | 56.44 | 74.7  | 93.56 | 75.91 | 62.42 | 74.75  | 21150.52  |
| DR-BERT-hn(full)   | 55.35 | 74.87 | 94.12 | 75.27 | 61.01 | 74.16  | 23.59  |
| DR-BERT(no-full,bert-fp-mono,epoch=5)      | 53.17 | 72.13  | 93.22 | 73.62 | 58.99 | 72.39  | 23.74  |
| DR-BERT(full=5,bert-fp,epoch=5)      | 54.09  |  72.65 | 93.12 | 74.14 | 59.8 | 72.98  | 23.8  |
| DR-BERT(full=5,bert-fp-mono,epoch=5)      | 55.58 | 74.58  | 93.49 | 75.44 | 61.52 | 74.38  | 23.62  |

<!-- tacl: token-aware contrastive learning for response selection -->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| DR-BERT(full=5,bert-fp-mono,epoch=5)       | 57.1  | 74.6  | 94.3   | 76.8 | 63.8 | 75.4  | 45.33  |
| DR-BERT-HN(q-q-bm25,full=5,bert-fp-mono,epoch=5)       | 56.35  | 75.1  | 93.78   | 76.24 | 62.98 | 75.04  | 22.71  |
| DR-BERT+TACL(full=5,bert-fp-mono,epoch=5)  | 57.98 | 72.5  | 92.25  | 76.43 | 64.19 | 75.06  | 45.33  |
| DR-BERT+TACL(full=5,bert-fp-mono,epoch=5)  | 57.93 | 73.17 | 92.79  | 76.62 | 64.39 | 75.19  | 45.73  |
| DR-BERT+TACL(full=5,bert-fp-mono,epoch=5,no-loss2)   | 58.28 | 73.54  | 93.06 | 77.07 | 64.79 | 75.43 | 45.04 |
| DR-BERT+TACL(full=5,bert-fp-mono,epoch=5,no-loss3,4) | 57.48 | 72.6   | 92.56 | 76.21 | 63.78 | 74.92 | 44.77 |
| DR-BERT+TACL(full=5,bert-fp-mono,epoch=5,no-loss2,no-pad) | 57.28 | 73.64   | 93.51 | 76.6 | 63.78 | 74.99 | 45.34 |


<!-- the hyper-paraeter batch size-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  | 14800.94  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  | 15135.32  |
| SA-BERT+BERT-FP    | 51.46 | 69.43  | 92.82 | 71.99 | 57.07 | 70.72  | 15135.32  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| DR-BERT(bsz=16,max_len=64/32,bert-fp-mono,full=5)    | 53.99 | 74.17 | 93.72 | 74.57| 59.8| 73.19| 23.55 |
| DR-BERT(bsz=32,max_len=64/32)    | 56.21 | 73.74 | 94.04 | 75.4 | 61.82 | 74.33| 23.79 |
| DR-BERT(bsz=48,max_len=64/32)    | 56.84 | 74.79 | 93.86 | 76.26|63.03 | 74.95 | 23.92 |
| DR-BERT(bsz=64,max_len=64/32)    | 55.46 | 76.76 | 94.13 |76.03 |61.62 | 74.77| 24.33 |
| DR-BERT(bsz=80,max_len=64/32)    | 58.08  | 75.86 | 94.42 | 77.18| 64.34 | 75.82 | 24.23 |
| DR-BERT(bsz=96,max_len=64/32)    | 56.46 | 75.95 | 94.44 | 76.19| 62.53| 75.12| 23.51 |
| DR-BERT(bsz=112,max_len=64/32)   | 56.54 | 74.6 | 94.59 | 75.9| 62.53| 74.8| 23.65 |
| DR-BERT(bsz=128,max_len=64/32)   | 56.21 | 74.71 | 94.56 | 75.77| 62.12| 74.7| 28.87 |
| DR-BERT(bsz=256,max_len=64/32)   | 56.59 | 74.99 | 93.9 | 76.27| 62.83| 74.98| 24.5 |
| DR-BERT(bsz=256,max_len=128/32,8*32 gather)   | 56.3  | 74.23 | 94.51 | 75.76 | 62.32 | 74.69 |  22.74 |
| DR-BERT(bsz=512,max_len=128/32,8*64 gather)   | 56.62 | 75.68 | 94.26 | 76.29 | 62.63 | 75.09 | 23.4 |
| DR-BERT(bsz=1024,max_len=128/32,8*128 gather) | 57.43 | 75.19 | 93.38 | 76.62 | 63.64 | 75.33 | 22.81 |
| DR-BERT-ext-neg(bsz=64,max_len=128/32,ext_neg_size=256)    | 57.49 | 75.57 | 93.53| 76.59 | 63.43 | 75.47  | 23.4  |


| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  | 14800.94  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  | 15135.32  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   | 21150.52  |
| poly-encoder(m=128)| 53.64 | 70.17 | 92.24 |  72.86 | 58.89 |  71.81  |  30397.72 |
| poly-encoder-full(m=128)| 55.42 | 74.2  | 92.9  | 75.02 | 62.02 | 74.51  | 26724.77  |
| poly-encoder-full+hn(m=128)| 56.13 | 75.25  | 94.23  | 75.94 | 62.12 | 74.77  | 28.44  |
| dual-bert-hn(simcse[256,10],epoch=1,lr=1e-5,grad=1,no-other-hn) | 59.76 | 74.89  | 93.81  | 77.71 | 65.86 | 76.52  | 25.91  |
| dual-bert-hn(bert-mask-da-dmr[0.4-0.6,aug_t=10],epoch=5,no-other-hn,cosine) | 58.47 | 73.59  | 94.23  | 76.72 | 64.44 | 75.44  | 22.8  |
| dual-bert-hn(bert-mask-da-dmr[0.4-0.6,aug_t=10],epoch=5,no-other-hn,cosine,no-full) | 54.92 | 71.85  | 93.38  | 74.28 | 60.71 | 73.18  | 22.8  |
| dual-bert-hn-bce | 14.43 | 26.14  | 58.7  | 36.46 | 16.26 | 35.78  | 23.22  |
| dual-bert+full(cosine scheduler)     | 56.54 | 75.12 | 94.18 | 76.04 | 62.53 | 74.78  | 24070.67  |
| dual-bert-ctx-cut+full    | 57.65 | 74.32 | 94.02 | 76.37 | 63.64 | 75.19  | 24070.67  |
| dual-bert+full-warmup(0.05)     | 56.54 | 75.12 | 94.18 | 76.04 | 62.53 | 74.78  | 24070.67  |
| dual-bert+full(bsz=128,max_len=128,32)     | 58.1 | 74.49 | 94.98 | 76.89 | 64.04 | 75.81  | 24010.91  |
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
| dual-bert-hn(simcse hard negative) | 55.94 | 74.38 | 93.75 | 75.58 | 62.02 | 74.43  | 23525.63  |
| dual-bert-hn(simcse hard negative,no last utterance candidates) | 56.78 | 74.13 | 94.61 | 76.12 | 62.93 | 75.0  | 23551.79  |

<!-- 500 size select test set-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   | Time Cost(ms) |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ | --------- |
| BERT               | 40.41 | 60.6  |  87.51| 63.9 | 45.27 | 62.49  |  15.54 |
| SA-BERT+BERT-FP    | 49.71 | 68.54 | 93.13 | 71.51 | 55.53 | 70.08  | 14.82  |
| BERT-FP            | 48.84 | 70.84 | 92.71 | 71.22 | 54.33|  70.17  |  14.87 |
| poly-encoder       | 51.78 | 70.81 | 92.54 | 72.92 | 57.75|  71.5  |  25.39 |
| DR-BERT(bsz=80,max_len=64/32,bert-fp-mono,full=5)  | 57.08  | 74.55 | 94.32| 76.75 | 63.78| 75.36| 19.25 |
| DR-BERT w/o. NDAP  | 53.34 | 69.42 | 93.06| 73.21 |59.15 | 71.99| 20.18 |
| DR-BERT w/o. FG    | 51.19 | 70.47 | 91.65| 72.62 |57.34 |71.22 | 19.65 |

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

### 5.1 Fine-grained test results

| Methods         | NDCG@3 | NDCG@5 |
| --------------- | ------ | ------ |
| BERT            | 0.625  | 0.714  |
| BERT-FP         | 0.609  | 0.709  |
| SA-BERT(BERT-FP)| 0.674  | 0.753  |
| dual-bert       | 0.686  | 0.767  |
| DR-BERT(full)   | 0.699  | 0.776  |
| DR-BERT(bert-fp-mono,full)   | 0.702  | 0.780  |
| DR-BERT(bert-fp-mono,full,epoch=10)   | 0.702  | 0.779  |
| DR-BERT(bert-fp-mono,max_len=64_32,full=5,epoch=5,bsz=96)   | 0.712  | 0.783  |
| DR-BERT-hn(bert-fp-mono,full)   | 0.708  | 0.780  |
| DR-BERT-triplet(full)   | 0.702  | 0.776  |
| DR-BERT-memory(full)   | 0.694  | 0.774  |
| DR-BERT-hn(full)| 0.706  | 0.778  |
| DR-BERT-hn(full,from-dual-bert)| 0.708  | 0.778  |
| DR-BERT-hn-ctx(full,from-dual-bert)| 0.704  | 0.778  |
