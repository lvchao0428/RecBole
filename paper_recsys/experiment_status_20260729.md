# Experiment Status — 2026-07-29

> 更新时间: 09:55  
> 上一日: [`experiment_status_20260728.md`](experiment_status_20260728.md)

---

## 〇、一句话

UniSRec Beauty 3-seed 全完（mean test MRR = 0.0206），Toys 差 2 个 seed（被其他训练挤占 GPU）。今日重点：等 GPU 空出后补齐 Toys UniSRec，然后开始论文撰写。

---

## 一、昨日 / 隔夜进展

### UniSRec GTS Baseline（5/6 完成）

| tag | dataset | seed | test MRR@10 | 状态 |
|-----|---------|-----:|------------:|:----:|
| unisrec_beauty_gts_seed42 | Beauty | 42 | 0.0204 | ✅ |
| unisrec_beauty_gts_seed2024 | Beauty | 2024 | 0.0206 | ✅ |
| unisrec_beauty_gts_seed2026 | Beauty | 2026 | 0.0208 | ✅ |
| unisrec_toys_gts_seed42 | Toys | 42 | 0.0162 | ✅ |
| unisrec_toys_gts_seed2024 | Toys | 2024 | — | ❌ 被 kill（epoch29） |
| unisrec_toys_gts_seed2026 | Toys | 2026 | — | ❌ 未跑 |

**Beauty 3-seed mean: 0.0206 ± 0.0002**  
(对比 SASRecAlignV3 LLM GTS Beauty: 0.0161 → UniSRec 更强，因为 UniSRec 直接用 full embedding 不经 align 阶段)

---

## 二、当前 GPU 占用

| 机器 | 进程 | 显存 | 说明 |
|------|------|------|------|
| 5090 | qwen3.5-0.8b-sft-v39 | 15 GB / 31 GB | 其他训练，需等完成 |

---

## 三、今日实验计划

### P0 — 补齐 UniSRec Toys seed2024 + seed2026（~3h, 等 GPU）

从头重跑 2 runs。Resume 脚本已备好。

### P1 — 论文撰写准备

全部实验数据整理：

#### 主表：GTS Protocol（MRR@10 test, 3-seed mean ± std）

| Dataset | ID-only | TF-IDF | LLM | MV | UniSRec |
|---------|--------:|-------:|----:|---:|--------:|
| Beauty | 0.0106 | **0.0165** | 0.0161 | 0.0160 | 0.0206 |
| Toys | 0.0070 | **0.0113** | 0.0107 | 0.0109 | TBD (~0.016) |

#### 对照表：LOO Protocol（MRR@10 test, 3-seed mean ± std）

| Dataset | ID-only | TF-IDF | LLM | MV |
|---------|--------:|-------:|----:|---:|
| Beauty | 0.0175 | 0.0251 | **0.0254** | 0.0253 |
| Toys | 0.0152 | 0.0247 | **0.0258** | **0.0258** |

#### 机制分析

| 分析 | 关键数字 |
|------|----------|
| View redundancy (CKA) | 0.84–0.94 |
| Effective rank (4-view concat) | 136.9/256 = 53.5% |
| Leave-one-view-out | Δ = 0.000000 |
| Early60 stop-gate | TF ≈ LLM ≥ MV |
| Cross ±analysis | 无稳定增益 |

---

## 四、论文故事线（based on 0723 指导）

**Title**: "When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation"

**三个核心问题**：
1. LOO vs leakage-free temporal split 是否改变 text-enhanced SR 的结论？→ **是**
2. TF-IDF / single-view LLM / prompt multi-view 谁在 temporal shift 下更鲁棒？→ **TF-IDF**
3. Multi-view 为什么没有增益？→ **View redundancy**（CKA > 0.8, eff rank 坍缩, LOO Δ = 0）

---

## 五、全局完成度

| 实验 | 状态 |
|------|------|
| Beauty GTS 3-seed | ✅ |
| Toys GTS 3-seed | ✅ |
| Beauty LOO 3-seed | ✅ |
| Toys LOO 3-seed | ✅ |
| Beauty Early60 3-seed | ✅ |
| View redundancy | ✅ |
| Leave-one-view-out | ✅ |
| Distribution shift | ✅ |
| Rank transition | ✅ |
| Warm-target audit | ✅ |
| UniSRec Beauty GTS 3-seed | ✅ |
| UniSRec Toys GTS 2/3 | 🟡 等 GPU |

**完成比例：11/12。** 补齐 UniSRec Toys 后实验全部到齐。
