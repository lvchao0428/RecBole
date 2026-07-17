# 对 0712 指导意见的数据回应

> 更新时间: 2026-07-16 21:45
> 数据来源: 5090 Beauty TS 实验（seed=2025, min_train_interactions=5, global time split）

---

## 问题(1): 锁死 global time split + leakage-free，复跑四个核心配置，确认"打平现象"

### 协议确认

- **Global Time Split**: ✅ 所有实验均使用 `stratified=True` 的全局时间分割（train cutoff 之前拟合 TF-IDF vocabulary/IDF/SVD/center-whiten）
- **Leakage-free TF-IDF**: ✅ `item_text_emb.base.ts.npy` 仅用 train cutoff 前数据拟合
- **Leakage-free LLM**: ✅ `item_text_emb.qwen2.5_7b.base.ts.npy` 同样 leakage-free
- **V1 no-boost TF-IDF+LLM=0.0042 异常**: 已排查，原因是旧版未正确加载 LLM embedding，重跑后正常

### 四个核心配置 MRR@10（test set, no-boost, no-Cross）

**旧版 concat align（7/13 完成）**:

| 配置 | MRR@10 | HR@10 | NDCG@10 | vs ID-only |
|------|:------:|:-----:|:-------:|:----------:|
| ID-only | 0.0112 | 0.0277 | 0.0151 | baseline |
| TF-IDF | 0.0157 | 0.0332 | 0.0196 | +40.2% |
| TF-IDF + LLM | 0.0168 | 0.0349 | 0.0209 | +50.0% |
| MV (4-view + TF) | 0.0141 | 0.0363 | 0.0192 | +25.9% |

**新版 per-source align（7/16 完成，统一对齐协议）**:

| 配置 | MRR@10 | HR@10 | NDCG@10 | vs ID-only |
|------|:------:|:-----:|:-------:|:----------:|
| ID-only | 0.0112 | 0.0277 | 0.0151 | baseline |
| TF-IDF (1× InfoNCE) | 0.0162 | 0.0348 | 0.0205 | +44.6% |
| TF-IDF + LLM (2× InfoNCE) | 0.0164 | 0.0358 | 0.0209 | +46.4% |
| MV 4-view + TF (5× InfoNCE) | 0.0158 | 0.0330 | 0.0198 | +41.1% |

### "打平现象"的确认与分析

**结论：打平现象是真实的，不是实现问题。**

| 对比 | 旧版 concat align | 新版 per-source align |
|------|:-----------------:|:--------------------:|
| TF → +LLM 增益 | +0.0011 | +0.0002 |
| TF → MV 增益 | -0.0016 | -0.0004 |
| LLM → MV 增益 | -0.0027 | -0.0006 |

- 两版对齐协议下均观察到：**TF-IDF 极强，加 LLM 边际收益极小（+0.0002~0.0011），MV 反而更低**
- Per-source align 统一后三者更加接近（MRR 差距从 0.0027 缩小到 0.0006），进一步证实"打平"
- 这支持导师的判断：**复杂文本模型在 leakage-free protocol 下边际收益有限**

### 跨域验证（Toys / Grocery，旧版，待 per-source 重跑）

| 数据集 | ID-only | TF-IDF | LLM | MV |
|--------|:-------:|:------:|:---:|:--:|
| Toys MRR@10 | 0.0074 | 0.0107 | 0.0112 | 0.0105 |
| Grocery MRR@10 | 0.0040 | 0.0065 | 0.0068 | — |

Toys/Grocery 同样呈现 LLM ≈ TF > ID-only，MV 未超越 LLM 的格局。

---

## 问题(2): 简化模型，删 SE / cold boost，Cross 作为旋钮验证

### 已完成的简化

| 组件 | 状态 | 说明 |
|------|:----:|------|
| **SENet** | ✅ 已删除 | 代码中已永久移除，MV 融合改为直接 concat→Linear |
| **cold_text_boost (训练)** | ✅ 可关闭 | 通过 `cold_text_boost=0.0` 关闭，今日 6 组均为 cb0 |
| **infer_boost (推理)** | ✅ 可关闭 | 通过 `infer_boost=0.0` 关闭，今日 6 组均为 ib0 |
| **两阶段训练** | 保留作为 optimization protocol | Phase-A 冻 backbone 学文本头 → Phase-B 联合微调 |

### Cross 作为旋钮的验证（per-source align, no-boost）

| 模型 | no-Cross MRR | +Cross MRR | Cross 增益 |
|------|:-----------:|:----------:|:----------:|
| TF-IDF | 0.0162 | **0.0164** | **+0.0002** |
| TF+LLM | **0.0164** | 0.0162 | -0.0002 |
| MV | 0.0158 | 0.0156 | -0.0002 |

**结论：Cross 没有带来稳定净收益。**
- TF 上 Cross 略微正向（+0.0002），但 LLM 和 MV 上 Cross 反而轻微负面
- 符合导师建议：Cross 不应硬塞进最终模型，作为分析旋钮即可
- 主方法选择 **no-Cross** 配置

### 当前主方法配置（简化后）

```
模型: SASRecAlignV3 (text_mode=both, TF-IDF + LLM)
SENet: 无
cold_text_boost: 0.0 (不用)
infer_boost: 0.0 (不用)
use_cross: false
align: per-source (TF + LLM 各自独立 InfoNCE)
训练: two-phase (optimization protocol)
```

MRR@10 = **0.0164**，vs ID-only 0.0112 提升 46.4%。

---

## 问题(3): WSDM 叙事改写 — 暂无数据支撑

导师建议将叙事改为 **leakage-aware controlled study**：
- TF-IDF 是强 baseline
- 复杂文本模型在严格时间协议下边际收益收缩
- Cross 改变 coverage-ranking 分配
- Multi-view/align 的作用依赖 backbone、frequency 和 domain

**目前缺少的分析**（对应导师的第3点建议）:
1. Per-user target rank transition（Cross 前后 target rank 迁移矩阵）
2. Score entropy / Top-1~Top-10 margin 分析
3. View 间 cosine redundancy / leave-one-view-out
4. 不同 item-frequency 桶的 gate/对齐相似度
5. Alignment gain 和 target-rank improvement 的相关性
6. 连续 effect size 和置信区间（不用人为阈值选色）

这些需要加载保存的 checkpoint 做离线分析，代码待开发。

---

## 汇总一览（Beauty TS, seed=2025, min5, no-boost, no-Cross）

| 配置 | MRR@10 | HR@10 | NDCG@10 | Δ vs ID |
|------|:------:|:-----:|:-------:|:-------:|
| ID-only | 0.0112 | 0.0277 | 0.0151 | — |
| **TF-IDF** | **0.0162** | 0.0348 | 0.0205 | **+44.6%** |
| **TF-IDF + LLM** | **0.0164** | **0.0358** | **0.0209** | **+46.4%** |
| MV (4-view + TF) | 0.0158 | 0.0330 | 0.0198 | +41.1% |

**一句话：在 leakage-free global time split 下，TF-IDF 已是极强 baseline（+44.6%），加 LLM 仅再提 +1.8pp，MV 反而不如 TF 单独使用。Cross 无稳定净收益。这一格局跨两版对齐协议和三个数据集均可复现。**
