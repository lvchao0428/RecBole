# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_beauty_mv_lr5e4_do0.3_seed2024`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 12001

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.016058** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.015725 | -0.000333 | +0.000333 |
| Drop view_1 (Function) | 0,2,3 | 0.016117 | +0.000059 | -0.000059 |
| Drop view_2 (Audience) | 0,1,3 | 0.015761 | -0.000297 | +0.000297 |
| Drop view_3 (Style) | 0,1,2 | 0.015623 | -0.000435 | +0.000435 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_3 | Style | +0.000435 |
| 2 | view_0 | Description | +0.000333 |
| 3 | view_2 | Audience | +0.000297 |
| 4 | view_1 | Function | -0.000059 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
