# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_toys_mv_lr5e4_do0.3_seed2026`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `gts_last_warm`, `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval users (warm): 3007  ·  all-last: 4873

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.004785** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.004462 | -0.000324 | +0.000324 |
| Drop view_1 (Function) | 0,2,3 | 0.004777 | -0.000008 | +0.000008 |
| Drop view_2 (Audience) | 0,1,3 | 0.004623 | -0.000162 | +0.000162 |
| Drop view_3 (Style) | 0,1,2 | 0.004654 | -0.000131 | +0.000131 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | +0.000324 |
| 2 | view_2 | Audience | +0.000162 |
| 3 | view_3 | Style | +0.000131 |
| 4 | view_1 | Function | +0.000008 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
