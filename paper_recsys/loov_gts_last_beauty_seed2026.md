# Leave-One-View-Out Analysis

Checkpoint: `saved/sg_beauty_mv_lr5e4_do0.3_seed2026`  

Method: **zero-mask** dropped view buffer (projection dims unchanged)  

Protocol: `min_train_interactions=5`  

Active view indices: `[0, 1, 2, 3]`  ·  eval targets: 12001

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.016935** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.016710 | -0.000226 | +0.000226 |
| Drop view_1 (Function) | 0,2,3 | 0.017045 | +0.000109 | -0.000109 |
| Drop view_2 (Audience) | 0,1,3 | 0.017101 | +0.000166 | -0.000166 |
| Drop view_3 (Style) | 0,1,2 | 0.016829 | -0.000106 | +0.000106 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | +0.000226 |
| 2 | view_3 | Style | +0.000106 |
| 3 | view_1 | Function | -0.000109 |
| 4 | view_2 | Audience | -0.000166 |

## Interpretation

- Dropping any single view barely changes MRR → **high redundancy** (consistent with high inter-view CKA and low concat effective rank).
