# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_beauty_mv_lr5e4_do0.3_seed42`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `gts_last_warm`, `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval users (warm): 3137  ·  all-last: 4520

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.006291** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.005549 | -0.000742 | +0.000742 |
| Drop view_1 (Function) | 0,2,3 | 0.005711 | -0.000579 | +0.000579 |
| Drop view_2 (Audience) | 0,1,3 | 0.006009 | -0.000282 | +0.000282 |
| Drop view_3 (Style) | 0,1,2 | 0.005750 | -0.000541 | +0.000541 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | +0.000742 |
| 2 | view_1 | Function | +0.000579 |
| 3 | view_3 | Style | +0.000541 |
| 4 | view_2 | Audience | +0.000282 |

## Interpretation

- All views have similar contribution → **redundant / interchangeable**.
