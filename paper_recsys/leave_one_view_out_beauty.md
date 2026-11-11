# Leave-One-View-Out Analysis

Checkpoint: `saved/sg_beauty_mv_lr5e4_do0.3_seed42`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 12001

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.015906** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.015845 | -0.000061 | +0.000061 |
| Drop view_1 (Function) | 0,2,3 | 0.016018 | +0.000112 | -0.000112 |
| Drop view_2 (Audience) | 0,1,3 | 0.015743 | -0.000163 | +0.000163 |
| Drop view_3 (Style) | 0,1,2 | 0.015564 | -0.000342 | +0.000342 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_3 | Style | +0.000342 |
| 2 | view_2 | Audience | +0.000163 |
| 3 | view_0 | Description | +0.000061 |
| 4 | view_1 | Function | -0.000112 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
