# Rank Transition Analysis

## Rank Transition Matrix (no-Cross → +Cross)

| no-Cross \ +Cross | 1 | 2-5 | 6-10 | 11-50 | >50 | Total |
|---|---|---|---|---|---|---|
| **1** | 15 | 9 | 0 | 1 | 0 | 25 |
| **2-5** | 8 | 33 | 2 | 5 | 3 | 51 |
| **6-10** | 0 | 3 | 8 | 7 | 3 | 21 |
| **11-50** | 0 | 3 | 9 | 33 | 27 | 72 |
| **>50** | 0 | 2 | 1 | 24 | 2804 | 2831 |

**Summary**: Improved 1429 (47.6%), Degraded 1528 (50.9%), Unchanged 43 (1.4%)

## Score Distribution Comparison

| Metric | no-Cross | +Cross | Δ |
|--------|:--------:|:------:|:--:|
| Score entropy (Top-10) | 0.9134 | 0.9216 | +0.0082 |
| Top1-Top10 margin | 1.3901 | 1.3028 | -0.0873 |

## Top-K Head / Tail Share (train-frequency buckets)

Fraction of recommended items in Top-K that fall into each train-frequency bucket (head≥10, mid 3–9, low 1–2, unseen=0). Higher head share ⇒ more popularity bias.

| K | Bucket | no-Cross | +Cross | Δ |
|:-:|:------:|:--------:|:------:|:--:|
| 5 | head | 0.678 | 0.666 | -0.012 |
| 5 | mid | 0.144 | 0.155 | +0.011 |
| 5 | low | 0.178 | 0.179 | +0.001 |
| 5 | unseen | 0.000 | 0.000 | +0.000 |
| 10 | head | 0.697 | 0.683 | -0.014 |
| 10 | mid | 0.146 | 0.158 | +0.011 |
| 10 | low | 0.157 | 0.160 | +0.002 |
| 10 | unseen | 0.000 | 0.000 | +0.000 |
| 20 | head | 0.711 | 0.697 | -0.013 |
| 20 | mid | 0.151 | 0.159 | +0.008 |
| 20 | low | 0.139 | 0.144 | +0.005 |
| 20 | unseen | 0.000 | 0.000 | +0.000 |

## By Target Item-Frequency Bucket

| Bucket | #Samples | NC→CR Improved | NC→CR Degraded | Mean Rank NC | Mean Rank CR |
|--------|:--------:|:--------------:|:--------------:|:------------:|:------------:|
| head | 882 | 386 (44%) | 471 (53%) | 19609.2 | 22030.6 |
| mid | 619 | 314 (51%) | 292 (47%) | 40499.1 | 42587.8 |
| low | 483 | 237 (49%) | 241 (50%) | 68164.2 | 75413.2 |
| unseen | 1016 | 492 (48%) | 524 (52%) | 160126.3 | 162457.6 |