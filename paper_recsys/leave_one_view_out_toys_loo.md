# Leave-One-View-Out Analysis

> **INVALID / 已废弃**：该版本通过修改 `text_view_indices` 做 ablation，
> 但模型前向实际读取初始化时固定的 `active_buffer_positions`，所以四个
> `Δ=0` 是 no-op 产生的假结果。不要引用本文件；有效方法必须使用 zero-mask。

Checkpoint: `saved/loo_toys_mv_lr5e4_do0.3_seed2026`

| Config | Views Used | MRR@10 | Δ vs Full | Contribution |
|--------|-----------|:------:|:---------:|:------------:|
| **Full (4 views)** | 0,1,2,3 | **0.000901** | — | — |
| Drop view_0 (Description) | 1,2,3 | 0.000901 | +0.000000 | -0.000000 |
| Drop view_1 (Function) | 0,2,3 | 0.000901 | +0.000000 | -0.000000 |
| Drop view_2 (Audience) | 0,1,3 | 0.000901 | +0.000000 | -0.000000 |
| Drop view_3 (Style) | 0,1,2 | 0.000901 | +0.000000 | -0.000000 |

## View Contribution Ranking

| Rank | View | Name | Contribution (drop hurts by) |
|:----:|:----:|------|:----------------------------:|
| 1 | view_0 | Description | -0.000000 |
| 2 | view_1 | Function | -0.000000 |
| 3 | view_2 | Audience | -0.000000 |
| 4 | view_3 | Style | -0.000000 |

## Interpretation

- All views have similar contribution → **high redundancy** (consistent with SVD diagnosis)