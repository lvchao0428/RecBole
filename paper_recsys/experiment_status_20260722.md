# Experiment Status — 2026-07-22

Overnight Early60 + Cross Top-K 已全部完成（5090，`00:09` 结束）。结果已同步到本地。

## 本地产物

| 文件 | 内容 |
|------|------|
| `paper_recsys/early60_stopgate_beauty.csv` | Early 窗口 3-seed + LLM±Cross |
| `logs/early60_overnight.log` | 完整过夜日志 + 排序摘要 |
| `paper_recsys/stopgate_3seed_beauty_metrics.csv` | Main 窗口 stop-gate（对照） |
| `paper_recsys/rank_transition_beauty_{llm,tf,mv}.{md,json}` | Cross Top-K head/tail share |

## P0 — Early temporal window（Beauty，TS `[0.6,0.1,0.3]`）

冻结 Main 超参：TF `5e-4/0.3`，LLM `1e-4/0.1`，MV `5e-4/0.3`；train-only early60 特征；no-boost / no-Cross。

### Test MRR@10（3 seeds: 2025 / 42 / 2024）

| Model | seed2025 | seed42 | seed2024 | mean ± std |
|-------|---------:|-------:|---------:|-----------:|
| TF | 0.0052 | 0.0056 | 0.0052 | **0.0053 ± 0.0002** |
| MV | 0.0051 | 0.0051 | 0.0053 | **0.0052 ± 0.0001** |
| LLM | 0.0051 | 0.0050 | 0.0047 | **0.0049 ± 0.0002** |

Valid 上三路接近（~0.032–0.034），TF 略高。

### 与 Main 窗口对照（Main test MRR，既有 stop-gate）

| Model | Main mean ± std | Early mean ± std |
|-------|----------------:|-----------------:|
| TF | 0.0165 ± 0.0008 | 0.0053 ± 0.0002 |
| LLM | 0.0161 ± 0.0005 | 0.0049 ± 0.0002 |
| MV (do=0.3) | 0.0160 ± 0.0008 | 0.0052 ± 0.0001 |

> Early test 绝对值更低，因 test 段更长/更早 cutoff；**比的是排序与 MV 是否超 1 std，不是绝对数值**。

### Stop-gate 判定

- Main：TF ≈ LLM ≥ MV，MV 未稳定超过单视图 1 std  
- Early：TF ≥ MV ≈ LLM（mean），MV **仍未**相对 TF/LLM 拉开 1 std  
- 两窗口一致结论：**MV 不作主方法；stop-gate 可正式触发**

说明：Early 单 seed 排序不完全固定（2024: MV>TF>LLM；2025: TF≈LLM≈MV；42: TF>MV>LLM），但 3-seed mean 与方差支持「无显著 MV 增益」。

## P1 — Cross 封板

### Top-K head share（Main ckpt，Top-10）

| Model | no-Cross | +Cross | Δ |
|-------|---------:|-------:|------:|
| LLM | 0.665 | 0.667 | +0.002 |
| TF | 0.697 | 0.683 | −0.014 |
| MV | 0.742 | 0.672 | −0.070 |

Cross **没有**稳定抬高 head 占比。

### Early LLM ±Cross（seed=2025）

| | valid MRR | test MRR |
|--|----------:|---------:|
| no-Cross | 0.0324 | 0.0051 |
| +Cross | 0.0320 | 0.0051 |

Early 上 Cross 无增益 → 支持「删默认」；窗口依赖也未显现正向收益。

## 可对老师说的一句话

严格 GTS + train-only 特征下，Beauty 在 Main 与 Early 两窗口上均为 **TF ≈ LLM ≥ MV**（MV 无稳定 ≥1 std 增益）；Cross 无稳定 MRR/NDCG 收益，Top-K head share 也无系统性恶化/改善叙事支持其作默认模块。WSDM 主线转为 leakage-aware controlled study：静态文本有效，prompt multi-view / Cross 为负结果与机制分析。
