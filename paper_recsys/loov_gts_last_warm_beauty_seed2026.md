# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_beauty_mv_lr5e4_do0.3_seed2026`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `gts_last_warm`, `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval users (warm): 3137  ·  all-last: 4520

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.006141** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.006173 | +0.000032 | -0.000032 |
| Drop view_1 (Function) | 0,2,3 | 0.005883 | -0.000258 | +0.000258 |
| Drop view_2 (Audience) | 0,1,3 | 0.005782 | -0.000359 | +0.000359 |
| Drop view_3 (Style) | 0,1,2 | 0.006041 | -0.000100 | +0.000100 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_2 | Audience | +0.000359 |
| 2 | view_1 | Function | +0.000258 |
| 3 | view_3 | Style | +0.000100 |
| 4 | view_0 | Description | -0.000032 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
