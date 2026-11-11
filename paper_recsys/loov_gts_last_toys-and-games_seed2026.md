# Leave-One-View-Out Analysis

Checkpoint: `saved/wrap_toys_mv_lr5e4_do0.3_seed2026`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 13186

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.011340** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.010857 | -0.000483 | +0.000483 |
| Drop view_1 (Function) | 0,2,3 | 0.011245 | -0.000095 | +0.000095 |
| Drop view_2 (Audience) | 0,1,3 | 0.011153 | -0.000187 | +0.000187 |
| Drop view_3 (Style) | 0,1,2 | 0.010816 | -0.000524 | +0.000524 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_3 | Style | +0.000524 |
| 2 | view_0 | Description | +0.000483 |
| 3 | view_2 | Audience | +0.000187 |
| 4 | view_1 | Function | +0.000095 |

## Interpretation

- All views have similar contribution → **redundant / interchangeable**.
