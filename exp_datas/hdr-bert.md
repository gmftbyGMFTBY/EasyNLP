# Restoration-200k Dataset

* Recall Performance

* Rerank performance

| Models             | R10@1 | R10@2 | R10@5 | MRR   | P@1 | MAP |
| ------------------ | ----- | ----- | ----- | ----- | --- | --- |
| dual-bert          | 53.55 | 73.26 | 93.18 | 74.47 | 59.76 | 73.09 |
| dual-bert-hier-trs | 48.34 | 65.56 | 90.27 | 69.26 | 53.92 | 67.96 |
| dual-bert-hier-trs-pos | 50.02 | 69.32 | 90.69 | 71.1 | 55.53 | 69.73 |
| dual-bert-hier-trs-gpt2 | 49.25 | 67.29 | 89.24 | 69.87 | 54.33 | 68.8 |
| dual-bert-hier-trs-bert-gpt2 | 50.12 | 69.58 | 88.8 | 70.96 | 55.73 | 69.86 |
| dual-bert-hier-trs-pos-mv(mv_num=3) | 50.77 | 67.82 | 90.66 | 71.13 | 56.14 | 69.84 |
| dual-bert-hier-trs-pos-mv-colbert(mv_num=3) | 51.63 | 68.66 | 91.31 | 71.84 | 57.14 | 70.68 |
