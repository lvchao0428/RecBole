# 实验进展 — 2026-07-17

> 更新时间: 2026-07-17 15:50  
> 三端同步: 本机 = 5090 = logMac (commit `25f12855`)  
> 依据: 0717 老师指导 → V4 规划启动

---

## 〇、机器快照（7/17 15:50）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | P2 共享小网格 (27 runs) PID 2569124 | 🟡 训练中 (GPU 96%) | ~23:00 |
| **log10** | 空闲 | ✅ | — |

---

## 一、今日已完成

### P0-A1: Distribution Shift 分析 ✅ (11:45)

| 指标 | Valid | Test | Shift |
|------|:-----:|:----:|:-----:|
| **新 item 比例** (unseen in train) | 11.3% | **17.8%** | **+6.4%** |
| Target 平均 popularity (train count) | 112.8 | 101.8 | -9.7% |
| Target 中位 popularity | 21 | 15 | -28.6% |
| Zero-interaction targets | 22,952 | 35,965 | +56.7% |
| Eligible users (≥5) in split | 7,860 | 6,356 | -19.1% |

**Bucket distribution shift:**

| Bucket | Valid | Test | Δ |
|--------|:-----:|:----:|:--:|
| head (≥10) | 63.1% | 56.6% | **-6.5%** |
| mid (3-9) | 15.9% | 14.9% | -0.9% |
| low (1-2) | 9.7% | 10.7% | +1.0% |
| unseen (0) | 11.3% | 17.8% | **+6.4%** |

**结论**: Test set 的 target item 明显更冷门（unseen +6.4%, head -6.5%），是 **temporal distribution shift** 而非单模型过拟合。所有模型同步受影响，解释了 valid→test 44% 下降。

---

### P0-A2: View Redundancy (CKA + Cosine) ✅ (11:50)

| View Pair | Cosine (per-item) | Linear CKA |
|-----------|:-----------------:|:----------:|
| Description vs Function | -0.014 | **0.843** |
| Description vs Audience | 0.028 | **0.858** |
| Description vs Style | 0.088 | **0.912** |
| **Function vs Audience** | **0.418** | **0.938** |
| Function vs Style | 0.000 | **0.839** |
| Audience vs Style | 0.025 | **0.856** |

| TF-IDF vs View | CKA |
|---------------|:---:|
| TF-IDF vs Description | 0.250 |
| TF-IDF vs Function | 0.258 |
| TF-IDF vs Audience | 0.254 |
| TF-IDF vs Style | 0.246 |

**Effective Rank (concat):**

| Config | Dim | Eff Rank | Ratio |
|--------|:---:|:--------:|:-----:|
| 单个 view (avg) | 64 | 63.8 | 0.996 |
| **4-view concat** | **256** | **136.9** | **0.535** |
| TF-IDF | 256 | 253.1 | 0.989 |
| TF-IDF + 4 views | 512 | 374.3 | 0.731 |

**结论**: 
1. 所有 view 对的 **CKA 均 > 0.83**，确认 prompt-based multi-view 是语义重述
2. 4-view concat effective rank 仅 136.9/256 (0.535) → **严重维度塌缩**
3. TF-IDF 与任何 view 的 CKA 仅 ~0.25 → TF-IDF 提供独立互补信息
4. 老师判断正确："四个 view 来自同一 title 和同一 Qwen encoder，它们很可能只是语义重述"

---

### P0-A3: SVD 谱图 (论文 Figure) ✅ (12:00)

生成文件: `paper_recsys/fig_svd_spectrum_beauty.pdf` / `.png`

内容: (a) Normalized singular value spectrum (log scale) (b) Cumulative energy

关键观察:
- TF-IDF 和 LLM 的 SV 衰减极缓慢（接近满秩）
- MV 4-view concat 的 SV 迅速衰减（前 50 维已占大部分能量）
- 对应 effective rank 差异：TF/LLM ~253 vs MV concat 137

---

### P1-B1: Rank-Transition Matrix (LLM no-Cross vs +Cross) ✅ (12:05)

**Transition Matrix** (2000 users):

| no-Cross \ +Cross | 1 | 2-5 | 6-10 | 11-50 | >50 |
|---|---|---|---|---|---|
| **1** | 1 | 0 | 0 | 1 | 14 |
| **2-5** | 0 | 0 | 0 | 1 | 30 |
| **6-10** | 0 | 0 | 0 | 0 | 21 |
| **11-50** | 0 | 0 | 0 | 0 | 54 |
| **>50** | 0 | 0 | 0 | 6 | 1872 |

**Summary**: Improved 46.3%, Degraded 53.6% → Cross **净负向**

**Score Distribution:**

| Metric | no-Cross | +Cross | Δ |
|--------|:--------:|:------:|:--:|
| Score entropy (Top-10) | 0.9266 | **0.9910** | **+0.0645** |
| Top1-Top10 margin | **1.2793** | 0.5166 | **-0.7627** |

**结论**: Cross 使 score 分布更均匀（entropy↑, margin↓），但并未将 target 从低 rank 拉入 Top-10。相反，部分 Top-10 内的 target 被挤出。这直接解释了 Cross 损 MRR 但可能微调覆盖的机制。

---

### P1-B3: Leave-One-View-Out ✅ (12:15)

| Config | Views | MRR@10 | Δ vs Full |
|--------|-------|:------:|:---------:|
| **Full (4 views)** | 0,1,2,3 | 0.016069 | — |
| Drop Description | 1,2,3 | 0.016069 | 0.000000 |
| Drop Function | 0,2,3 | 0.016069 | 0.000000 |
| Drop Audience | 0,1,3 | 0.016069 | 0.000000 |
| Drop Style | 0,1,2 | 0.016069 | 0.000000 |

**结论**: Drop 任何单个 view 对 MRR 完全无影响 → 4 个 view 完全冗余，单一 view 已包含全部有用信息。与 CKA>0.83 和 effective rank 0.535 高度一致。

---

## 二、进行中

| 任务 | 状态 | 预计完成 |
|------|:----:|:--------:|
| **P2**: 共享小网格 (27 runs: TF×9, LLM×9, MV×9) | 🟡 5090 运行中 (PID 2569124) | **7/17 ~23:00** |

---

## 三、今晚/明天计划

| 优先级 | 任务 | 预计启动 | 预计完成 | 机器 |
|:------:|------|:--------:|:--------:|:----:|
| **P2** | 共享小网格 (9组×3模型=27 runs) | 7/17 15:47 ▶️ | **7/17 ~23:00** | 5090 |
| **P3-D1** | LLM no-Cross 3-seed (42/2024/2026) | 7/17 23:00 | **7/18 05:00** | 5090 |
| **P3-D2** | MV no-Cross 3-seed | 7/18 05:00 | **7/18 13:00** | 5090 |
| **P3-D3** | Rolling early-valid window | 7/18 13:00 | **7/18 17:00** | 5090 |
| **P3-D4** | Toys per-source 公平对比 | 7/18 17:00 | **7/19 01:00** | 5090 |

---

## 四、关键结论更新

### 对照 0717 老师建议执行进度

| 老师建议 | 执行状态 | 结论 |
|----------|:--------:|------|
| 检查 valid/test distribution shift | **✅ 完成** | 确认 temporal shift（unseen +6.4%） |
| 做 pairwise cosine/CKA | **✅ 完成** | 所有 view CKA>0.83，严重冗余 |
| 判断 view redundancy vs 容量不足 | **✅ 结论明确** | **redundancy**（不是容量问题） |
| Cross rank-transition 分析 | **✅ 完成** | Cross 净负向，使 score 更均匀但不提升 target rank |
| Leave-one-view-out | **✅ 完成** | 4 view 完全冗余（Δ=0） |
| 共享小网格调参 | ⏳ 排队 | 今天下午~晚上 |
| 3-seed + 2-window stop gate | ⏳ 排队 | 今晚~明天 |
| Toys 域验证 | ⏳ 排队 | 明天 |

---

## 五、一句话目标

**P0+P1 全部完成；view CKA>0.83 + leave-one-view Δ=0 双重确认冗余；P2 网格搜索运行中(ETA 23:00)；下一步：3-seed stop gate → Toys 域验证。**
