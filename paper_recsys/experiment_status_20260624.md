# 5090 实验状态梳理（2026-06-24）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-24 20:18（5090 实查）  
> **GPU 状态**: RTX 5090 **空闲**（util 0%，无训练进程）  
> **当前主线**: BC phaseb50 ✅ · Grocery seed=2024 ✅ → **Grocery multiseed 待定（Go）**

**相关文档**:
- 结果汇总: [`experiment_summary_20260624.md`](experiment_summary_20260624.md)
- BC 分层: [`book_crossing_stratified_results_20260623.md`](book_crossing_stratified_results_20260623.md)
- Grocery 规划: [`grocery_experiment_plan_20260623.md`](grocery_experiment_plan_20260623.md)
- Food 对照: [`experiment_status_20260623.md`](experiment_status_20260623.md)

---

## 1. 进度总览

```
[✅] BC phaseb50 四配置 seed=2024     6/24 09:26 完成
[✅] Grocery 流水线 RUN_ID=20260623_grocery  6/24 17:59 完成
[⏸] Grocery multiseed（Go 条件满足，未启动）
[✅] Food 四配置                       6/23 09:38 完成
```

| 实验 | RUN_ID | 完成时间 | 日志 |
|------|--------|----------|------|
| BC phaseb50 | `20260623_bc_four_phaseb50` | 6/24 09:26 | `logs/queue_bc_four_phaseb50.log` |
| Grocery | `20260623_grocery` | 6/24 17:59 | `logs/grocery_pipeline_history.log` |
| Food | `20260622_food` | 6/23 09:38 | `logs/5090_food_20260622_food_history.log` |

---

## 2. book-crossing phaseb50（seed=2024 · test @10）

| Model | MRR@10 | HR@10 | vs ID (MRR) | MRR new/few/freq |
|-------|--------|-------|-------------|------------------|
| **ID** | **1.95%** | 3.87% | — | 1.09 / 2.04 / 3.93 |
| TF-IDF | 2.20% | 3.65% | **+13%** | 1.42 / 2.47 / 4.26 |
| TF-IDF+LLM | 2.20% | 3.71% | **+13%** | 1.38 / 2.45 / 4.29 |
| **MV** | **2.27%** | 3.87% | **+16%** | 1.42 / 2.44 / 4.48 |

**run_metrics**: ID `20260623-175744` · TF-IDF `20260623-204501` · LLM `20260624-002001` · MV `20260624-092622`

**结论**: 四配置全部完成；MRR 阶梯 ID < TF-IDF ≈ LLM < MV（边际），增益 +13~16%，仍远低于 Beauty +45%。**appendix 定性不变**。

---

## 3. Grocery 第三域（seed=2024 · test @10）

| Model | MRR@10 | HR@10 | vs ID (MRR) | vs ID (HR) | MRR new/few/freq |
|-------|--------|-------|-------------|------------|------------------|
| **ID** | **2.08%** | **6.36%** | — | — | 0.62 / 1.18 / 3.73 |
| TF-IDF | 2.93% | 5.77% | **+41%** | −9.3% | 0.85 / 1.80 / 5.20 |
| TF-IDF+LLM | 2.95% | 5.83% | **+42%** | −8.3% | 0.88 / 1.77 / 5.26 |
| **MV** | **3.01%** | 6.02% | **+45%** | −5.3% | 0.86 / 1.78 / 5.38 |

**Stratified HR@10**: ID new/few/freq = **1.56 / 3.33 / 11.67** · TF-IDF = 1.32 / 3.12 / 10.57 · MV = 1.32 / **3.19** / 11.09

**vs Beauty（4-seed mean）**: MRR 增益形态一致（+41~45%）；Overall HR 上 Grocery 文本低于 ID（Beauty 上 TF-IDF/MV **提升** HR），属已知 MRR–HR trade-off，freq 层 MV 最接近 ID。

**run_metrics**: ID `20260624-131307`（5090 fallback）· TF-IDF `20260624-140726` · LLM `20260624-151509` · MV `20260624-175942`

**流水线耗时**（6/24）:

| Step | 耗时 | 备注 |
|------|------|------|
| 1_emb | 3.4h | TF-IDF + Qwen 1v/4v |
| 2_id_fallback | 20min | log10 wait 失败，5090 重跑 ID |
| 3_tfidf | 54min | |
| 4_llm | 68min | |
| 5_mv | 2.7h | |

**Go/No-Go 判断**（对标 Beauty 主表）:

| 指标 | 目标 | Grocery 实测 | 判定 |
|------|------|--------------|------|
| TF-IDF / ID (MRR) | ≥1.30× | **1.41×** (2.93/2.08) | ✅ **Go** |
| MV > TF-IDF+LLM | 显著 > | 3.01% > 2.95% | ✅ **Go** |
| MV MRR 增益 | ~Beauty +45% | **+45%** vs ID | ✅ |

→ **建议启动 Grocery multiseed**（2024/2025/2026/42，四配置）。

---

## 4. Food 对照域（已完成 · 6/23）

| 配置 | test MRR@10 | test HR@10 | vs ID |
|------|-------------|------------|-------|
| ID | 0.60% | 2.00% | — |
| TF-IDF | 0.68% | 1.70% | +13% |
| LLM | 0.65% | 1.30% | +8% |
| MV | 0.65% | 1.46% | +8% |

非 Amazon tail-heavy 对照；overall ID > 文本（HR），不作第三域主表。

---

## 5. 已知问题

| 项 | 说明 |
|----|------|
| ID seed=2020 | BC / Grocery ID 的 run_metrics 中 yaml 覆盖 CLI `--seed 2024` |
| log10 wait 失败 | Grocery pipeline 在 5090 重跑 ID（结果与 log10 一致 MRR≈2.08%） |
| Grocery RUN_ID | 实际为 `20260623_grocery`（非规划中的 `20260624_grocery`） |

---

## 6. 下一步

1. **Grocery multiseed** — 4 seeds × 4 configs（主表第三域）
2. BC — 不必 multiseed（appendix 已有 seed2025 历史 + seed2024 phaseb50）
3. 可选：修复 baseline yaml 硬编码 seed 后重跑 ID（低优先级）

**监控**（当前无任务）:

```bash
ssh charlie@www.ultrapp.online 'pgrep -af run_5090; nvidia-smi'
```
