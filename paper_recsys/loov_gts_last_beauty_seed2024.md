# Leave-One-View-Out Analysis

Checkpoint: `saved/sg_beauty_mv_lr5e4_do0.3_seed2024`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 12001

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.015289** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.014933 | -0.000356 | +0.000356 |
| Drop view_1 (Function) | 0,2,3 | 0.015221 | -0.000068 | +0.000068 |
| Drop view_2 (Audience) | 0,1,3 | 0.015237 | -0.000052 | +0.000052 |
| Drop view_3 (Style) | 0,1,2 | 0.015215 | -0.000074 | +0.000074 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | +0.000356 |
| 2 | view_3 | Style | +0.000074 |
| 3 | view_1 | Function | +0.000068 |
| 4 | view_2 | Audience | +0.000052 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
