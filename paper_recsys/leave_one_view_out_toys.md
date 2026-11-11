# Leave-One-View-Out Analysis

Checkpoint: `saved/wrap_toys_mv_lr5e4_do0.3_seed42`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 13186

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.010175** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.009982 | -0.000193 | +0.000193 |
| Drop view_1 (Function) | 0,2,3 | 0.010067 | -0.000108 | +0.000108 |
| Drop view_2 (Audience) | 0,1,3 | 0.010051 | -0.000124 | +0.000124 |
| Drop view_3 (Style) | 0,1,2 | 0.010025 | -0.000150 | +0.000150 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | +0.000193 |
| 2 | view_3 | Style | +0.000150 |
| 3 | view_2 | Audience | +0.000124 |
| 4 | view_1 | Function | +0.000108 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
