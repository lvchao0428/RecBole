# Leave-One-View-Out Analysis

Checkpoint: `saved/ps_beauty_mv_nc_noboost_seed2025`

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.016069** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.016069 | +0.000000 | -0.000000 |
| Drop view_1 (Function) | 0,2,3 | 0.016069 | +0.000000 | -0.000000 |
| Drop view_2 (Audience) | 0,1,3 | 0.016069 | +0.000000 | -0.000000 |
| Drop view_3 (Style) | 0,1,2 | 0.016069 | +0.000000 | -0.000000 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | -0.000000 |
| 2 | view_1 | Function | -0.000000 |
| 3 | view_2 | Audience | -0.000000 |
| 4 | view_3 | Style | -0.000000 |

## Interpretation

- All views have similar contribution → **high redundancy** (consistent with SVD diagnosis)