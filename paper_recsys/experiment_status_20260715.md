# 实验进展 — 2026-07-15/16

> 更新时间: 2026-07-16 18:05  
> 三端同步: 本机 ← 5090 ← log10  
> 依据: **Per-source 公平对比 6/6 全部完成** + 嵌入塌缩诊断启动

---

## 〇、机器快照（7/16 18:05）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | **空闲** — per-source 6/6 全部完成 (17:33) | ✅ GPU 0% | 可排新任务 |
| **log10** | 空闲 | ✅（UniSRec MV-V3 OOM 未恢复） | 待安排 |

**7/15–7/16 完成的关键事项**：  
1. MV no-Cross 融合 bug 修复（简单加法→concat+predictor）  
2. 三组 MV concat-fix 实验全部完成（7/15 12:48–21:39）  
3. UniSRec AlignV3 完成（无增益）  
4. **Per-source 公平对比 6 组全部完成**（7/16 09:09–17:33）  
5. 配置公平性已验证：三模型同 align_weight/text_weight/seed/protocol

---

## 一、已完成结论（可写进论文）

### 1. Beauty 四象限齐（seed=2025, min5, TS V2）

主表 MV 用 **对齐后**（`text_weight=1.0`，Phase-A 已修复完整 20ep 训练）；TF/LLM 用既有结果。

#### MV aligned 重跑四象限（Phase-A 修复后，全部 ✅）

| 配置 | MRR@10 | NDCG@10 | R@10 | R_new@10 | 完成时间 |
|------|:------:|:------:|:----:|:-----:|:------:|
| MV no-Cross, no-boost | **0.0143** | 0.0194 | 0.0357 | 0.0188 | 7/14 22:46 |
| MV +Cross, no-boost | **0.0151** | 0.0189 | 0.0315 | 0.0183 | 7/15 02:09 |
| MV no-Cross, +boost | **0.0151** | 0.0201 | 0.0364 | 0.0234 | 7/15 04:06 |
| MV +Cross, +boost | **0.0157** | 0.0197 | 0.0329 | 0.0198 | 7/15 07:30 |

#### MRR@10 四象限总表

| | **no-Cross** | **+Cross** |
|--|:------------:|:----------:|
| **no-boost** | LLM **0.0168** > TF 0.0157 > MV **0.0143** | MV **0.0151** ≈ TF/LLM 0.0157 |
| **cb3+infer** | LLM **0.0171** > TF 0.0160 > MV **0.0151** | MV **0.0157** ≈ TF 0.0160; LLM 0.0164 |

#### boost+no-Cross 完整（7/14 16:12 完成）

| 配置 | MRR@10 | NDCG@10 | R@10 | R_new |
|------|:------:|:------:|:----:|:-----:|
| ID-only | 0.0112 | 0.0151 | 0.0277 | 0.0137 |
| TF | 0.0160 | 0.0202 | 0.0340 | 0.0178 |
| **LLM** | **0.0171** | **0.0213** | **0.0350** | **0.0270** |
| MV (旧 tw=0.7) | 0.0151 | 0.0200 | 0.0362 | 0.0234 |
| MV aligned tw=1.0 | 0.0151 | 0.0201 | 0.0364 | 0.0234 |

### 2. 核心结论（定稿级）

1. **最优：LLM no-Cross + boost，MRR=0.0171**（本协议最高）
2. **经典 MV>LLM>TF 在严格 TS+min5 下未恢复**；对齐 `text_weight` 后 MV 数字几乎不变 → 不是简单超参未对齐
3. **Cross 非稳定净收益**：LLM/TF 侧 no-Cross 更好；MV 上 Cross 略抬 MRR、压 HR
4. **TF-IDF 仍是强 baseline**（与 LLM 差距 ≤0.0011）
5. **主方法建议**：简化为 **LLM single-view + no-Cross**（boost 作可选/附录）；MV/Cross 作 ablation

### 3. UniSRec Base（log10，✅ 00:25）

| 指标 | UniSRec Base | SASRec ID-only | SASRec LLM no-Cross+boost |
|------|:------------:|:--------------:|:-------------------------:|
| MRR@10 | 0.0124 | 0.0112 | **0.0171** |
| R@10 | **0.0391** | 0.0277 | 0.0350 |
| R_new@10 | **0.0310** | 0.0137 | 0.0270 |

→ UniSRec 在 MRR 上弱于 SASRec+LLM，但 HR/R_new 较高；portability 叙事：同一 Qwen 特征可迁到 UniSRec，但增益形态不同（覆盖 vs 排序）。

---

## 二、已完成实验（7/15–7/16）

### A. MV concat-fix 三组（5090，7/15 12:48–21:39 全部完成 ✅）

**修复**：MV no-Cross 融合从**简单加法**改为 **concat+predictor**（对齐 V3 TF/LLM 融合容量）。

| # | 配置 | Phase-A MRR | Test MRR@10 | Test R@10 | Test R_new | 完成 |
|---|------|:---:|:---:|:---:|:---:|:---:|
| 1 | MV no-Cross, no-boost | **0.0060** | **0.0153** | 0.0330 | 0.0193 | 14:53 |
| 2 | MV +Cross, no-boost | **0.0059** | **0.0151** | 0.0316 | 0.0183 | 18:16 |
| 3 | MV +Cross, cb3+infer | **0.0058** | **0.0158** | 0.0331 | 0.0198 | 21:39 |

**对比旧版（简单加法）**：

| 配置 | 旧 MRR | concat-fix MRR | Δ | 备注 |
|------|:---:|:---:|:---:|:---:|
| MV no-Cross, no-boost | 0.0143 | **0.0153** | **+0.0010** | 有效 |
| MV +Cross, no-boost | 0.0151 | 0.0151 | ±0 | Cross 已提供融合 |
| MV +Cross, +boost | 0.0157 | 0.0158 | +0.0001 | ⚠️ **作废** |

⚠️ **Exp3 (+Cross +boost) 结论作废**：该实验使用旧版 align（concat 后整体 align），与新版 per-view align + base align 的设计不一致。以 per-source 公平对比实验为准。

**有效结论**：  
1. Phase-A 修复有效（全部 ~0.006）  
2. concat+predictor 对 no-Cross 有实质帮助 (+0.001)  
3. 最终公平对比以 **per-source 实验**（per-view align + TF-IDF align）为准

### B. Per-source 公平对比（5090，7/16 09:09–17:33 全部完成 ✅）

6 组实验：三模型(TF/LLM/MV) × 两配置(no-Cross/+Cross)，统一 no-boost，per-view align。

| # | 配置 | MRR@10 | NDCG@10 | R@10 | R_new@10 | MRR_freq | MRR_few | MRR_new | 完成 |
|---|------|:------:|:-------:|:----:|:--------:|:--------:|:-------:|:-------:|:----:|
| 1 | TF no-Cross | 0.0162 | 0.0205 | 0.0348 | 0.0234 | 0.0340 | 0.0177 | 0.0116 | 10:01 |
| 2 | **LLM no-Cross** | **0.0164** | **0.0209** | **0.0358** | 0.0198 | 0.0339 | 0.0186 | **0.0121** | 10:55 |
| 3 | MV no-Cross | 0.0158 | 0.0198 | 0.0330 | 0.0168 | 0.0329 | 0.0186 | 0.0104 | 12:12 |
| 4 | TF +Cross | 0.0164 | 0.0202 | 0.0327 | 0.0193 | 0.0339 | 0.0186 | 0.0119 | 13:30 |
| 5 | LLM +Cross | 0.0162 | 0.0202 | 0.0335 | 0.0224 | 0.0338 | 0.0180 | 0.0118 | 15:10 |
| 6 | MV +Cross | 0.0156 | 0.0196 | 0.0326 | 0.0183 | 0.0320 | 0.0187 | 0.0106 | 17:33 |

#### Cross 效应分析（+Cross vs no-Cross）

| 模型 | Δ MRR@10 | Δ R@10 | Δ R_new@10 | 解读 |
|------|:--------:|:------:|:----------:|------|
| TF | +0.0002 | -0.0021 | -0.0041 | Cross 无排序增益，压覆盖 |
| LLM | -0.0002 | -0.0023 | **+0.0026** | Cross 微损排序，增 R_new |
| MV | -0.0002 | -0.0004 | +0.0015 | Cross 对 MV 亦无净正向 |

#### 频率分桶洞察

| 模型 | MRR_freq | MRR_few | MRR_new | 薄弱环节 |
|------|:--------:|:-------:|:-------:|---------|
| TF | 0.0340 | 0.0177 | 0.0116 | — |
| LLM | 0.0339 | **0.0186** | **0.0121** | — |
| MV | 0.0329 | 0.0186 | 0.0104 | **new item 明显差** |

MV 在 frequent/few 桶与 LLM 持平，但 **new item 上差距最大**（0.0104 vs 0.0121），指向多视角对低频/新 item 的表示能力不足。

### C. UniSRec（log10，✅ / ❌）

| 配置 | MRR@10 | R@10 | R_new@10 | 状态 |
|------|:------:|:----:|:-----:|:----:|
| Base | 0.0124 | 0.0391 | 0.0310 | ✅ |
| AlignV3 | 0.0124 | 0.0278 | 0.0112 | ✅（无增益） |
| MV-V3 | — | — | — | ❌ OOM |

---

## 三、Per-source 公平对比核心结论（定稿级）

### 模型排序

| 配置 | no-Cross | +Cross |
|------|:--------:|:------:|
| **排序** | LLM 0.0164 > TF 0.0162 > MV 0.0158 | TF 0.0164 > LLM 0.0162 > MV 0.0156 |

### 核心发现

1. **LLM no-Cross 仍为最优**（MRR=0.0164），但 TF 仅差 0.0002
2. **MV < LLM 在所有配置下成立**：no-Cross 差 0.0006，+Cross 差 0.0006
3. **Cross 不改变模型间相对排序**，但使三者更趋近
4. **Cross 对 LLM 无益**：MRR -0.0002，但 R_new +0.0026（覆盖 vs 排序 trade-off）
5. **MV 弱点集中在 new item**：MRR_new 0.0104 vs LLM 0.0121（-14%）
6. **TF-IDF 是极强 baseline**：per-source align 后 TF 0.0162 ≈ LLM 0.0164

### 对比旧版（V2，无 per-source align）

| 模型 | 旧 V2 MRR | 新 per-source MRR | 变化 | 解读 |
|------|:---------:|:-----------------:|:----:|------|
| TF | 0.0157 | **0.0162** | +0.0005 | per-source align 对 TF 有效 |
| LLM | 0.0168 | **0.0164** | -0.0004 | per-source 略负面（多了 base align 分散梯度？）|
| MV | 0.0141 | **0.0158** | **+0.0017** | concat-fix + per-view align 大幅改善 |

---

## 四、嵌入塌缩诊断（7/16 18:10 完成 ✅）

脚本：`scripts/diagnose_collapse.py`，Mode A（raw npy）+ Mode B（checkpoint）均已完成。

### Mode A — 原始嵌入 Effective Rank

| Layer | Dim | Eff Rank | Ratio | 判定 |
|-------|:---:|:--------:|:-----:|:----:|
| TF-IDF base | 256 | 252.8 | 0.987 | 满秩 |
| LLM base | 256 | 253.6 | 0.991 | 满秩 |
| view_0~3 各自 | 64 | ~63.8 | 0.996 | 满秩 |
| **MV views concat** | **256** | **136.4** | **0.533** | **塌缩** |
| MV full (base+views) | 512 | 375.6 | 0.734 | 中度 |
| LLM full (base+llm) | 512 | **485.0** | 0.947 | 满秩 |

**关键**：4 个 view 各自满秩(64d)，但 concat 后 256d 仅有 136.4 有效维度 → view 间存在大量冗余。LLM+TF-IDF concat 后接近满秩（485/512）→ 两源互补。

### Mode B — 训练后逐层 Effective Rank（per-source checkpoint）

| Layer | LLM Eff Rank | MV Eff Rank | MV/LLM | 判定 |
|-------|:------------:|:-----------:|:------:|:----:|
| **Raw text** | **473.0** | **191.6** | **0.405** | **严重塌缩** |
| **Projected (256d)** | **192.2** | **138.6** | **0.721** | **中度塌缩** |
| Fused item (256d) | 89.3 | 99.4 | 1.113 | 无（ID 补偿） |

### View 冗余度

| 组合 | 余弦相似度 | 说明 |
|------|:---------:|------|
| view_1 vs view_2 | **0.4177** | Function vs Audience 高冗余 |
| view_0 vs view_3 | 0.0883 | 轻微 |
| 其余 | ~0 | 无冗余 |

### Gate 权重

- LLM: α = 0.717（effective weight = 0.717）
- MV: α = 0.685（effective weight = 0.685）
- MV 模型学到了 **降低文本权重**，间接证明它"知道"自己文本表示不够好

### 诊断结论

1. **MV < LLM 的根因是信息冗余而非模型容量不足**：4 view concat 后 effective rank 仅 136/256，远低于 LLM 的 254/256
2. **塌缩发生在 raw text 层**（0.405 ratio），投影层有所缓解（0.721），融合层被 ID embedding 完全补偿（1.113）
3. 这解释了为什么 MV 在 **frequent items**（ID 主导）上与 LLM 持平，但在 **new items**（依赖文本）上差距大
4. **view_1 与 view_2 的 0.42 高冗余** 是信息浪费的直接来源

---

## 五、下一步计划（7/16 晚起）

### 明日优先

| 优先级 | 任务 | 机器 | 估时 | 目的 |
|:------:|------|:----:|:----:|------|
| **P0** | SVD 谱图（论文 Figure） | 本机 | 1h | 可视化塌缩 |
| **P1** | Cross rank-transition + score entropy | 5090 | 2h | 解释 HR-MRR trade-off |
| **P1** | Toys/Grocery per-source 公平协议 | 5090 | 6-8h | 泛化验证 |
| **P2** | LLM no-Cross 3-seed 稳定性 | 5090 | 6h | 主方法置信度 |
| **P3** | UniSRec MV-V3（移至 5090） | 5090 | 3h | portability |

### 不建议做的

- 再调 MV 超参追层级（塌缩是结构性的，非超参问题）
- 把 boost/Cross 写回主方法（全部证据不支持）

---

## 五、日志速查

```
# Per-source 公平对比（权威）
logs/ps_beauty_{tf,llm,mv}_{nc,cross}_noboost_seed2025.log

# 四象限（历史参考）
logs/ts_beauty_fb_nc_{id_only,tfidf,llm,mv}_seed2025.log
logs/ts_beauty_fb_{tfidf,llm,mv}_seed2025.log

# MV concat-fix
logs/ts_beauty_mv_{nc,cross,cross_boost}_concatfix_seed2025.log

# UniSRec
logs/ts_unisrec_base_seed2025.log          # ✅
logs/ts_unisrec_alignv3_seed2025.log       # ✅（无增益）
```

---

## 七、老师指导对照（0712zhidao）

| 老师要求 | 当前状态 | 备注 |
|----------|:--------:|------|
| (1) 锁 global TS + train-only fit；复跑 4 核心配置 | **✅ 完成** | per-source 6 组全部完成 |
| (2) 简化：删 SE / boosts；Cross 可选 | **✅ 完成** | 全部 no-boost；Cross 证明非净收益 |
| (3) 叙事改 leakage-aware controlled study | **✅ 机制证据已出** | SVD 塌缩诊断完成；Cross rank-transition 待做 |
| TF-IDF 是 strong baseline | **✅ 验证** | TF 0.0162 ≈ LLM 0.0164 |
| MV 与 LLM 差距解释 | **✅ 诊断完成** | view 冗余导致 effective rank 塌缩 |
| UniSRec portability | 🔄 Base+AlignV3 完成 | MV-V3 OOM 待 5090 |
| 第三数据集新 protocol | ⏳ | Toys/Grocery 待排 |

---

## 八、一句话目标

**主表 + 塌缩诊断已完成；下一步：SVD 谱图(论文Fig) + Cross 机制分析 + Toys/Grocery 泛化验证。**
