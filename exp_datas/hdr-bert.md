# Restoration-200k Dataset

* Recall Performance

* Rerank performance

| Models             | R10@1 | R10@2 | R10@5 | MRR   | P@1 | MAP |
| ------------------ | ----- | ----- | ----- | ----- | --- | --- |
| dual-bert          | 53.55 | 73.26 | 93.18 | 74.47 | 59.76 | 73.09 |
| dual-bert-hier-trs | 48.34 | 65.56 | 90.27 | 69.26 | 53.92 | 67.96 |
| dual-bert-hier-trs-pos | 50.02 | 69.32 | 90.69 | 71.1 | 55.53 | 69.73 |
| dual-bert-hier-trs-gpt2 | 49.25 | 67.29 | 89.24 | 69.87 | 54.33 | 68.8 |
| dual-bert-hier-trs-bert-gpt2(nlayer=3) | 50.12 | 69.58 | 88.8 | 70.96 | 55.73 | 69.86 |
| dual-bert-hier-trs-bert-mv-gpt2(nlayer=3, mv=10) | 50.62 | 68.63 | 90.24 |71.25  | 56.14 | 69.95 |
| dual-bert-hier-trs-bert-mv-gpt2(nlayer=6, mv=10) | 50.12 | 69.23 | 91.67 | 71.35  |55.73 | 70.07 |
| dual-bert-hier-trs-pos-mv(mv_num=3) | 50.77 | 67.82 | 90.66 | 71.13 | 56.14 | 69.84 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=3) | 51.63 | 68.66 | 91.31 | 71.84 | 57.14 | 70.68 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=2,nlayer=3,colbert,no-ext-ndap) | 49.98 | 67.91  |92.24 | 70.98 | 55.33 | 69.76 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=5,nlayer=3) | 52.8 | 71.01 | 92.71 | 73.38 | 58.55 | 72.28 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=3) | 54.66 | 71.58 | 92.51 | 74.36 | 60.16 | 73.02 |
| dual-bert-hier-trs-pos-mv-colbert(ctx_res_mv_num=5,nlayer=3,no-ext-ndap) | 52.3 | 70.69 | 92.71 | 72.92 | 58.35 | 71.55 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=3,no-ext-ndap) | 52.52 | 69.53 | 92.24 | 72.86 | 58.15 | 71.67 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=3,no-ext-ndap) | 52.75 | 70.88 | 93.39 | 73.37 | 58.35 | 72.03 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=3,no-ext-ndap,bsz=128) | 53.42 | 68.56| 93.73 | 73.5| 59.56 | 72.13 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=15,nlayer=3,no-ext-ndap,bsz=128) | 53.12 | 70.64 | 92.45 | 73.38 | 58.75 | 72.14 |
| dual-bert-hier-trs-pos-mv-colbertcls-res(mv_num=10,nlayer=3,no-ext-ndap) | 50.39 | 69.87 | 92.91 | 71.84 | 55.94 | 70.67 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=2) | 54.48 | 71.18 | 92.54 | 74.29 | 60.56 | 72.89 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=10,nlayer=1) | 51.53 | 71.43 | 92.59 | 72.77 | 57.14 | 71.54 |
| dual-bert-hier-trs-pos-mv-colbert-gpt2(mv_num=10,nlayer=3) | 49.75 | 68.61 | 93.04 | 71.46 |55.53  | 70.25 |
