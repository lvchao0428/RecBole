# Rank Transition Analysis

## Rank Transition Matrix (no-Cross → +Cross)

| no-Cross \ +Cross | 1 | 2-5 | 6-10 | 11-50 | >50 | Total |
|---|---|---|---|---|---|---|
| **1** | 19 | 7 | 1 | 0 | 0 | 27 |
| **2-5** | 8 | 21 | 7 | 3 | 4 | 43 |
| **6-10** | 1 | 8 | 7 | 8 | 2 | 26 |
| **11-50** | 0 | 3 | 5 | 43 | 25 | 76 |
| **>50** | 1 | 1 | 1 | 18 | 2807 | 2828 |

**Summary**: Improved 1487 (49.6%), Degraded 1475 (49.2%), Unchanged 38 (1.3%)

## Score Distribution Comparison

| Metric | no-Cross | +Cross | Δ |
|--------|:--------:|:------:|:--:|
| Score entropy (Top-10) | 0.9195 | 0.9209 | +0.0014 |
| Top1-Top10 margin | 1.3379 | 1.2975 | -0.0403 |

## Top-K Head / Tail Share (train-frequency buckets)

Fraction of recommended items in Top-K that fall into each train-frequency bucket (head≥10, mid 3–9, low 1–2, unseen=0). Higher head share ⇒ more popularity bias.

| K | Bucket | no-Cross | +Cross | Δ |
|:-:|:------:|:--------:|:------:|:--:|
| 5 | head | 0.643 | 0.654 | +0.011 |
| 5 | mid | 0.173 | 0.172 | -0.001 |
| 5 | low | 0.184 | 0.174 | -0.010 |
| 5 | unseen | 0.000 | 0.000 | +0.000 |
| 10 | head | 0.665 | 0.667 | +0.002 |
| 10 | mid | 0.172 | 0.172 | -0.000 |
| 10 | low | 0.162 | 0.161 | -0.002 |
| 10 | unseen | 0.000 | 0.000 | +0.000 |
| 20 | head | 0.681 | 0.683 | +0.002 |
| 20 | mid | 0.173 | 0.172 | -0.001 |
| 20 | low | 0.146 | 0.145 | -0.001 |
| 20 | unseen | 0.000 | 0.000 | +0.000 |

## By Target Item-Frequency Bucket

| Bucket | #Samples | NC→CR Improved | NC→CR Degraded | Mean Rank NC | Mean Rank CR |
|--------|:--------:|:--------------:|:--------------:|:------------:|:------------:|
| head | 882 | 408 (46%) | 453 (51%) | 18863.4 | 22589.8 |
| mid | 619 | 281 (45%) | 326 (53%) | 34930.8 | 40225.2 |
| low | 483 | 223 (46%) | 255 (53%) | 65529.1 | 75738.2 |
| unseen | 1016 | 575 (57%) | 441 (43%) | 178780.8 | 169527.0 |