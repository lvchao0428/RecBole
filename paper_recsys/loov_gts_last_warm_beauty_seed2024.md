# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_beauty_mv_lr5e4_do0.3_seed2024`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `gts_last_warm`, `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval users (warm): 3137  ·  all-last: 4520

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.006640** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.006732 | +0.000091 | -0.000091 |
| Drop view_1 (Function) | 0,2,3 | 0.006569 | -0.000071 | +0.000071 |
| Drop view_2 (Audience) | 0,1,3 | 0.006479 | -0.000161 | +0.000161 |
| Drop view_3 (Style) | 0,1,2 | 0.006312 | -0.000329 | +0.000329 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_3 | Style | +0.000329 |
| 2 | view_2 | Audience | +0.000161 |
| 3 | view_1 | Function | +0.000071 |
| 4 | view_0 | Description | -0.000091 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
