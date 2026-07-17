# Rank Transition Analysis

## Rank Transition Matrix (no-Cross → +Cross)

| no-Cross \ +Cross | 1 | 2-5 | 6-10 | 11-50 | >50 | Total |
|---|---|---|---|---|---|---|
| **1** | 1 | 0 | 0 | 1 | 14 | 16 |
| **2-5** | 0 | 0 | 0 | 1 | 30 | 31 |
| **6-10** | 0 | 0 | 0 | 0 | 21 | 21 |
| **11-50** | 0 | 0 | 0 | 0 | 54 | 54 |
| **>50** | 0 | 0 | 0 | 6 | 1872 | 1878 |

**Summary**: Improved 926 (46.3%), Degraded 1073 (53.6%), Unchanged 1 (0.1%)

## Score Distribution Comparison

| Metric | no-Cross | +Cross | Δ |
|--------|:--------:|:------:|:--:|
| Score entropy (Top-10) | 0.9266 | 0.9910 | +0.0645 |
| Top1-Top10 margin | 1.2793 | 0.5166 | -0.7627 |