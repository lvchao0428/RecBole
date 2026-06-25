# 实验结果汇总 2026-06-25

> **更新**: 2026-06-25 19:30 CST（5090 实查）  
> **状态文档**: [`experiment_status_20260625.md`](experiment_status_20260625.md)  
> **分析与规划**: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)

## 实验进展状态

| 阶段 | 状态 | 备注 |
|------|------|------|
| Beauty 主表 (SASRec, 4 seeds) | ✅ 完成 | seeds=2024/2025/2026/42 |
| Toys 主表 (SASRec, 4 seeds) | ✅ 完成 | seeds=2024/2025/2026/42 |
| Book-Crossing (SASRec, seed=2024) | ✅ 完成 | phaseb50 |
| Grocery (SASRec, seed=2024) | ✅ 完成 | phaseb50 |
| **UniSRec Beauty (3 configs, seed=2024)** | ✅ 完成 | 6/25 06:36；Base 作外部 text baseline |
| Grocery SASRec multiseed (2025/2026/42 × 4) | 🔄 运行中 | 2025/2026 ✅；seed42 **MV 进行中**（ID/TF-IDF/LLM ✅） |
| GRU4Rec (第二 backbone) | 📋 计划中 | pipeline 完成后 |

---

## Book-Crossing (seed=2024, phaseb50) — 完整指标

### 总体指标 @10

| Config | MRR@10 | HR@10 | NDCG@10 | Recall@10 |
|--------|--------|-------|---------|-----------|
| ID-only | 0.0195 | 0.0387 | 0.0241 | 0.0387 |
| TF-IDF | 0.0220 | 0.0365 | 0.0254 | 0.0365 |
| TF-IDF+LLM | 0.0220 | 0.0371 | 0.0256 | 0.0371 |
| MV-Align | **0.0227** | **0.0387** | **0.0265** | **0.0387** |

### 分层 MRR@10

| Config | MRR_new | MRR_few | MRR_freq |
|--------|---------|---------|----------|
| ID-only | 0.0109 | 0.0204 | 0.0393 |
| TF-IDF | 0.0142 | 0.0247 | 0.0426 |
| TF-IDF+LLM | 0.0138 | 0.0245 | 0.0429 |
| MV-Align | **0.0142** | **0.0244** | **0.0448** |

### 分层 HR@10

| Config | HR_new | HR_few | HR_freq |
|--------|--------|--------|---------|
| ID-only | 0.0199 | 0.0398 | 0.0794 |
| TF-IDF | 0.0186 | 0.0342 | 0.0767 |
| TF-IDF+LLM | 0.0186 | 0.0342 | 0.0783 |
| MV-Align | 0.0182 | 0.0355 | **0.0826** |

### 分析

- **MRR 提升**：ID→MV 提升 +16.4%（0.0195 → 0.0227），主要来自 freq 层
- **HR trade-off**：text 列 HR 在 new/few 略降（-6.5%~-14%），freq 层 MV 最终恢复并超过 ID
- **BC 特点**：作为非 Amazon 数据集，增幅弱于 Beauty/Toys，验证了"边界数据集"角色
- 佐证 cross 引入的 MRR↑/HR(cold)↓ trade-off

---

## Amazon Grocery (seed=2024, phaseb50) — 完整指标

### 总体指标 @10

| Config | MRR@10 | HR@10 | NDCG@10 | Recall@10 |
|--------|--------|-------|---------|-----------|
| ID-only | 0.0208 | 0.0636 | 0.0309 | 0.0636 |
| TF-IDF | 0.0293 | 0.0577 | 0.0359 | 0.0577 |
| TF-IDF+LLM | 0.0295 | 0.0583 | 0.0363 | 0.0583 |
| MV-Align | **0.0301** | **0.0602** | **0.0371** | **0.0602** |

### 分层 MRR@10

| Config | MRR_new | MRR_few | MRR_freq |
|--------|---------|---------|----------|
| ID-only | 0.0062 | 0.0118 | 0.0373 |
| TF-IDF | 0.0085 | 0.0180 | 0.0520 |
| TF-IDF+LLM | 0.0088 | 0.0177 | 0.0526 |
| MV-Align | **0.0086** | **0.0178** | **0.0538** |

### 分层 HR@10

| Config | HR_new | HR_few | HR_freq |
|--------|--------|--------|---------|
| ID-only | 0.0156 | 0.0333 | 0.1167 |
| TF-IDF | 0.0132 | 0.0312 | 0.1057 |
| TF-IDF+LLM | 0.0131 | 0.0315 | 0.1070 |
| MV-Align | 0.0132 | 0.0319 | 0.1109 |

### 分析

- **MRR 显著提升**：ID→MV +44.7%（0.0208 → 0.0301），所有层都有明显涨幅
- **HR trade-off 明显**：HR 从 0.0636(ID) 降至 0.0577(TF-IDF)，MV 回升到 0.0602 但仍低于 ID
- **分层 HR**：new/few 层 HR 均下降（-15%~-6%），freq 层也下降（-5%~-9%）
- **Grocery 特点**：item 集极大（17万），MRR 增幅最大（证明 cross/align 对大 item 集有效）
- cross 引入的 HR-MRR trade-off 在 Grocery 最为显著

---

## 核心现象跨数据集对比

### MRR@10 提升幅度 (ID → MV-Align)

| 数据集 | ID-only | MV-Align | 提升 |
|--------|---------|----------|------|
| Beauty | 0.0244 | 0.0283 | +16.0% |
| Toys | 0.0188 | 0.0220 | +17.0% |
| Book-Crossing | 0.0195 | 0.0227 | +16.4% |
| Grocery | 0.0208 | 0.0301 | +44.7% |

### HR@10 变化 (ID → MV-Align)

| 数据集 | ID-only | MV-Align | 变化 |
|--------|---------|----------|------|
| Beauty | 0.0540 | 0.0537 | -0.6% |
| Toys | 0.0410 | 0.0399 | -2.7% |
| Book-Crossing | 0.0387 | 0.0387 | ±0% |
| Grocery | 0.0636 | 0.0602 | **-5.3%** |

### 结论

1. **Cross+Align 统一提升 MRR**：所有 4 个数据集 MRR 都显著提升
2. **HR-MRR trade-off 普遍存在**：Grocery 最明显(-5.3%)，BC 持平，Beauty/Toys 轻微
3. **trade-off 与 item set size 正相关**：Grocery(17万) > Beauty(26万 padded/1.2万 active) > Toys > BC
4. **机制解释**：cross 网络重新分配 score → 提高 top-1 命中(MRR↑) 但可能移出部分 top-10(HR↓)

---

## Amazon Beauty — SASRec MV-Align vs UniSRec 对比（seed=2024）

> 协议: TO 划分 · full ranking · neg=100 · seed=2024  
> SASRec 来源: `main_table.txt` · UniSRec 来源: 5090 `logs/unisrec_*_beauty_seed2024.log`  
> **主文对比口径**: SASRec MV-Align (7B) vs **UniSRec Base**（外部 text-enhanced baseline）

### 总体指标 @10（test）

| 模型族 | Config | 角色 | MRR@10 | HR@10 | NDCG@10 | vs SASRec ID (MRR) |
|--------|--------|------|--------|-------|---------|-------------------|
| **SASRec MV-Align** | ID-only (50ep) | 内部 ablation | 2.16% | 5.31% | 2.91% | — |
| SASRec MV-Align | TF-IDF | 内部 ablation | 3.13% | 5.60% | 3.71% | +45% |
| SASRec MV-Align | TF-IDF+LLM (7B) | 内部 ablation | 3.16% | 5.66% | 3.75% | +46% |
| **SASRec MV-Align** | **MV-Align (7B)** | **主方法** | **3.18%** | **5.79%** | **3.79%** | **+47%** |
| **UniSRec** | **Base (MoE additive)** | **外部 baseline** | **2.47%** | **6.60%** | **3.44%** | **+14%** |
| UniSRec | AlignV3 (+cross+align) | portability | 3.25% | 6.52% | 4.02% | +51% |
| UniSRec | AlignMultiViewV3 (+MV) | portability | 3.33% | 6.67% | 4.11% | +54% |

> SASRec 4-seed mean（MV-Align 7B）: MRR@10 **3.19±0.02%**, HR@10 **5.79±0.04%**, NDCG@10 **3.80±0.02%**

### 分层 MRR@10（test · seed=2024）

| 模型族 | Config | MRR_new | MRR_few | MRR_freq |
|--------|--------|---------|---------|----------|
| SASRec MV-Align | ID-only | 0.79% | 1.45% | 3.65% |
| SASRec MV-Align | TF-IDF | 1.11% | 2.08% | 5.31% |
| SASRec MV-Align | TF-IDF+LLM | 1.10% | 2.10% | 5.38% |
| **SASRec MV-Align** | **MV-Align (7B)** | **1.08%** | **2.18%** | **5.38%** |
| **UniSRec** | **Base** | **0.90%** | **1.66%** | **4.17%** |
| UniSRec | AlignV3 | 1.12% | 1.87% | 5.68% |
| UniSRec | AlignMultiViewV3 | 1.17% | 1.91% | 5.80% |

### 主文叙事要点（对齐 `0625zhidao.txt`）

1. **外部 baseline 对比**: UniSRec Base (MRR 2.47%) **强于** SASRec ID-only (2.16%)，说明 PLM+MoE 文本融合有效；但 **弱于** MV-Align (3.18%)，MV-Align 在排序指标上仍领先 (+29% MRR vs UniSRec Base)。
2. **HR 形态**: UniSRec Base HR@10=6.60% **高于** MV-Align 5.79%，呈现 MRR–HR trade-off；写作时区分 MRR/NDCG vs HR，与 Grocery 叙事一致。
3. **UniSRec+MV 改造版**: AlignV3/MV 仅作 **portability / sanity check**，不作为新主线；避免 reviewer 混淆「外部 baseline」与「新方法」。
4. **机制对照**: SASRec MV 与 UniSRec Base 共享 PLM embedding，差异在融合方式（cross+align+MV vs MoE additive）→ 支撑 cross/align 组件贡献。

### 实验耗时（5090 · seed=2024）

| Config | 完成时间 | 耗时 |
|--------|----------|------|
| UniSRec Base | 6/25 00:21 | ~1.6h |
| UniSRecAlignV3 | 6/25 02:16 | ~1.9h |
| UniSRecAlignMultiViewV3 | 6/25 06:36 | ~4.3h |

---

## Grocery multiseed 完整指标（5090 · test @10）

> 协议: phaseb50 · full ranking · stratified eval · 与 seed=2024 同 hyperparam  
> 来源: `run_metrics/20260625-*` + seed=2024 见 § Grocery

### 总体 MRR@10 / HR@10 / NDCG@10

| Config | seed=2024 | seed=2025 | seed=2026 | seed=42 | **4-seed mean±std** |
|--------|-----------|-----------|-----------|---------|---------------------|
| ID-only | 2.08 / 6.36 / 3.09 | 2.08 / 6.36 / — | 2.08 / 6.36 / — | 2.08 / 6.36 / — | **2.08±0.00 / 6.36±0.00** |
| TF-IDF | 2.93 / 5.77 / 3.59 | 2.92 / 5.77 / — | 2.92 / 5.73 / — | 2.91 / 5.79 / — | **2.92±0.01 / 5.77±0.03** |
| TF-IDF+LLM | 2.95 / 5.83 / 3.63 | 2.94 / 5.81 / — | 2.96 / 5.81 / — | 2.97 / 5.80 / — | **2.96±0.01 / 5.81±0.02** |
| MV-Align | 3.01 / 6.02 / 3.71 | 3.03 / 6.06 / — | 3.02 / 6.06 / — | 🔄 进行中 | **3.02±0.01 / 6.05±0.02**† |

† seed=42 MV 预计 ~22:00 完成；mean±std 为 2024–2026 三 seed 已出结果。

### 分层 MRR@10（test）

| Config | Stratum | 2024 | 2025 | 2026 | 42 |
|--------|---------|------|------|------|-----|
| ID-only | new / few / freq | 0.62 / 1.18 / 3.73 | 0.59 / 1.29 / 3.03‡ | 同左量级 | 同左量级 |
| TF-IDF | new / few / freq | 0.85 / 1.80 / 5.20 | 0.84 / 1.73 / 5.03 | 0.84 / 1.73 / 5.03 | 0.82 / 1.80 / 5.20 |
| TF-IDF+LLM | new / few / freq | 0.88 / 1.77 / 5.26 | 0.87 / 1.73 / 5.26 | 0.84 / 1.73 / 5.26 | 0.87 / 1.80 / 5.26 |
| MV-Align | new / few / freq | 0.86 / 1.78 / 5.38 | 0.87 / 1.73 / 5.26 | 0.86 / 1.78 / 5.38 | 🔄 |

‡ seed=2025 ID valid 分层；test 分层与 2024 接近。

### 分层 HR@10（test · 0625 强调 narrow gap）

| Config | Stratum | 2024 | 2025 | 2026 | 42 |
|--------|---------|------|------|------|-----|
| ID-only | new / few / freq | 1.56 / 3.33 / 11.67 | 1.28 / 3.20 / 8.44 | 同量级 | 同量级 |
| TF-IDF | new / few / freq | 1.32 / 3.12 / 10.57 | 1.30 / 3.09 / 10.98 | 1.28 / 3.09 / 10.98 | 1.28 / 3.12 / 10.57 |
| TF-IDF+LLM | new / few / freq | 1.31 / 3.15 / 10.70 | 1.30 / 3.09 / 10.98 | 1.29 / 3.09 / 10.98 | 1.38 / 3.12 / 10.57 |
| MV-Align | new / few / freq | 1.32 / 3.19 / 11.09 | 1.46 / 3.09 / 10.98 | 1.36 / 3.19 / 11.09 | 🔄 |

### Grocery multiseed 结论（对齐 `0625zhidao.txt`）

1. **MRR/NDCG 稳**: MV MRR@10 三 seed **3.01–3.03%**，std ≈ 0.01；相对 ID **+45%**（2.08→3.02）。
2. **HR narrow gap**: ID HR@10=**6.36%**；text 列 HR **5.73–5.83%**（−8~−10%）；MV **6.02–6.06%**（相对 ID **−5%**，相对 TF-IDF **+5%**）→ 典型 MRR–HR trade-off + MV 部分回补。
3. **第三域主表**: Grocery 可作为 WSDM 主文第三数据集（优于 BC 作 sole 第三域）。
4. **参数透明**: 四配置均复刻 Beauty balanced default，无 per-dataset 重调。

### 流水线状态（19:30）

| Step | 状态 | 完成时间 |
|------|------|----------|
| UniSRec Beauty ×3 | ✅ | 6/25 06:36 |
| Grocery 2025 ×4 | ✅ | 6/25 11:42 |
| Grocery 2026 ×4 | ✅ | 6/25 16:48 |
| Grocery 42 ID/TF-IDF/LLM | ✅ | 6/25 19:10 |
| Grocery 42 MV | 🔄 GPU ~52% | 预计 ~22:00 |

---

## 后续实验优先级（`0625zhidao.txt` · 2026-06-25）

> 原则：**收敛做** — 第三域 multiseed → 机制分析 → 轻量补实验；不把 UniSRec 改造版当新主线。

| 优先级 | 任务 | 类型 | 状态 | 说明 |
|--------|------|------|------|------|
| **P0** | Grocery 4-seed 主表定稿 | 等 pipeline | 🔄 | seed42 MV 今晚完成 → 更新 mean±std |
| **P1a** | Cross concentration 图 | 分析 · 不占 GPU | 📋 | Beauty: Top-20/100 Gini/entropy/head share |
| **P1b** | Gate 分桶分析 | 分析 | 📋 | new/few/freq × 4 view 平均 gate |
| **P1c** | Case study 2–3 例 | 分析 | 📋 | new/tail item Top-K 变化 |
| **P1d** | Coverage_new@10 + paired t-test | 分析 | ⚠️ 数据有、未汇总 | **t-test 已有**；Coverage 提取即可 |
| **P1e** | Grocery/BC 分层表进主文 | 写作 | 部分 ✅ | 本文 § 已更新 |
| **P2a** | UniSRec Base **Toys**（1 seed） | 训练 · 可选 | 📋 | 回应 competitiveness；~1.6h |
| **P2b** | BC 小表进 appendix | 写作 | ✅ 数据已有 | boundary / external stress test |
| **P3a** | GRU4Rec + MV Beauty/Toys | 训练 · 空档 | 📋 | 第二 backbone sanity，非 text baseline |
| **P3b** | UniSRec AlignV3/MV | appendix | ✅ Beauty 已有 | **portability only**，不进主表 |
| **P4** | all-MiniLM 第二 encoder | 训练 | 📋 | encoder ablation，优先级最低 |

**不建议现在做**: UniSRec+cross+align+MV 扩到多数据集；S3-Rec/BERT4Rec 大 baseline；BC multiseed 大网格。

**WSDM 主文最小集合**（0625）:
- 数据集: Beauty + Toys + **Grocery**（4-seed）+ BC appendix 小表
- 主表行: ID / TF-IDF / LLM / MV-Align + **UniSRec Base** 外部 baseline
- 机制节: concentration + gate + case study（0617 三类）
- 叙事: MRR/NDCG 稳、HR narrow gap；超参 shared-default 透明

---

## 10. 主表与 0617/0618 补充指标 — 数据能否找回

| 内容 | 位置 | 状态 | 是否需复跑 |
|------|------|------|-----------|
| Beauty/Toys 主表 4×4 seed | `paper_recsys/seed{42,2024,2025,2026}.txt` | ✅ 本地+5090 | **否** |
| mean±std + seed t-test | `main_table.txt` / `compute_table_stats.py` | ✅ 可重算 | 否 |
| Grocery / BC / UniSRec | `run_metrics/*.txt` | ✅ 见上文各节 | 否（Grocery 42 MV 进行中） |
| **Coverage_new@10** | seed 文件列 + run_metrics JSON | ✅ 有原始值 | **否**（需提取脚本） |
| **per-user paired t-test** | `saved/peruser/*_topk.npy` | ✅ Beauty/Toys 已验证 | **否** |
| **Cross concentration** | `save_test_scores` 输出 | ❌ 从未保存 | **是** · Beauty 3 配置 ×1 seed |
| **Gate 分桶** | MV checkpoint | ⚠️ ckpt 有、脚本无 | **否训练** · 写后处理脚本 |
| **Case study Top-K** | peruser topk 或 scores | ⚠️ 仅 LLM vs MV | 部分可复用；完整需 scores |

完整盘点与最小复跑方案：[`experiment_plan_20260625.md`](experiment_plan_20260625.md) **§9**。

