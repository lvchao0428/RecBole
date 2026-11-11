# Leave-One-View-Out Analysis

Checkpoint: `/home/charlie/project/RecBole/saved/wrap_toys_mv_lr5e4_do0.3_seed2024`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `gts_last_warm`, `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval users (warm): 3007  ·  all-last: 4873

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.004187** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.003895 | -0.000292 | +0.000292 |
| Drop view_1 (Function) | 0,2,3 | 0.004112 | -0.000075 | +0.000075 |
| Drop view_2 (Audience) | 0,1,3 | 0.004259 | +0.000072 | -0.000072 |
| Drop view_3 (Style) | 0,1,2 | 0.003702 | -0.000485 | +0.000485 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_3 | Style | +0.000485 |
| 2 | view_0 | Description | +0.000292 |
| 3 | view_1 | Function | +0.000075 |
| 4 | view_2 | Audience | -0.000072 |

## Interpretation

- Most important: view_3 (Style)
- Least important: view_2 (Audience)
