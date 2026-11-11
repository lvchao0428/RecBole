# Rank Transition Analysis

## Rank Transition Matrix (no-Cross → +Cross)

| no-Cross \ +Cross | 1 | 2-5 | 6-10 | 11-50 | >50 | Total |
|---|---|---|---|---|---|---|
| **1** | 21 | 7 | 0 | 1 | 0 | 29 |
| **2-5** | 12 | 22 | 2 | 6 | 0 | 42 |
| **6-10** | 0 | 1 | 10 | 6 | 3 | 20 |
| **11-50** | 1 | 5 | 4 | 40 | 21 | 71 |
| **>50** | 0 | 1 | 4 | 24 | 2809 | 2838 |

**Summary**: Improved 1546 (51.5%), Degraded 1418 (47.3%), Unchanged 36 (1.2%)

## Score Distribution Comparison

| Metric | no-Cross | +Cross | Δ |
|--------|:--------:|:------:|:--:|
| Score entropy (Top-10) | 0.9325 | 0.9227 | -0.0098 |
| Top1-Top10 margin | 1.2168 | 1.2821 | +0.0653 |

## Top-K Head / Tail Share (train-frequency buckets)

Fraction of recommended items in Top-K that fall into each train-frequency bucket (head≥10, mid 3–9, low 1–2, unseen=0). Higher head share ⇒ more popularity bias.

| K | Bucket | no-Cross | +Cross | Δ |
|:-:|:------:|:--------:|:------:|:--:|
| 5 | head | 0.721 | 0.656 | -0.065 |
| 5 | mid | 0.124 | 0.158 | +0.035 |
| 5 | low | 0.155 | 0.185 | +0.030 |
| 5 | unseen | 0.000 | 0.000 | +0.000 |
| 10 | head | 0.742 | 0.672 | -0.070 |
| 10 | mid | 0.127 | 0.165 | +0.038 |
| 10 | low | 0.131 | 0.163 | +0.032 |
| 10 | unseen | 0.000 | 0.000 | +0.000 |
| 20 | head | 0.755 | 0.685 | -0.070 |
| 20 | mid | 0.131 | 0.166 | +0.036 |
| 20 | low | 0.114 | 0.148 | +0.035 |
| 20 | unseen | 0.000 | 0.000 | +0.000 |

## By Target Item-Frequency Bucket

| Bucket | #Samples | NC→CR Improved | NC→CR Degraded | Mean Rank NC | Mean Rank CR |
|--------|:--------:|:--------------:|:--------------:|:------------:|:------------:|
| head | 882 | 394 (45%) | 469 (53%) | 15924.7 | 24099.2 |
| mid | 619 | 284 (46%) | 321 (52%) | 35301.5 | 38651.8 |
| low | 483 | 246 (51%) | 234 (48%) | 64653.5 | 67182.5 |
| unseen | 1016 | 622 (61%) | 394 (39%) | 178325.9 | 159538.0 |