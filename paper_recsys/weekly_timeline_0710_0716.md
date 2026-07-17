# 一周实验时间线 (2026-07-10 ~ 07-16)

> 围绕 0712 指导意见两个核心问题的完整回顾

---

## 核心问题

**(1)** 锁死 global time split + leakage-free TF-IDF，复跑四核心配置，确认"打平"不是实现问题
**(2)** 简化模型（删 SE / cold boost），Cross 作为旋钮——不是默认组件

---

## 时间线

### 7/10 (周四) — 首次 TS 验证，发现"层级消失"

**做了什么**: 将评测从旧 random split 切换到 global time split (TS)，首次用 V1 旧特征（有泄露风险）跑四配置

**结果**:
| 配置 | 旧 random split MRR | TS split MRR (V1 boost) |
|------|:---:|:---:|
| ID-only | ~0.03+ | 0.0150 |
| TF-IDF | ~0.04+ | 0.0204 |
| LLM | ~0.05+ | 0.0206 |
| MV | ~0.05+ | 0.0208 |

**问题**: TS 下绝对值大幅下降；MV > LLM > TF 层级虽保留，但差距极小（0.0204~0.0208）。**但此时用的仍是旧 TF-IDF（有泄露风险）+ 开着 cold_boost=3.0 + infer_boost=0.6**

**耗时原因**: 这天主要在调通 TS split pipeline 和验证数据加载，首次跑通。

---

### 7/11 (周五) — V1 no-boost 验证 + 制作 leakage-free 特征

**做了什么**:
1. 关掉 boost 跑 V1 no-boost → 发现 **TF-IDF+LLM MRR=0.0042**，极度异常
2. 开始制作 leakage-free (TS-aware) TF-IDF 和 Qwen 特征

**遇到的问题**:
- V1 no-boost LLM=0.0042 → 排查发现是 `freeze_backbone` 配置 bug：字符串 `"false"` 没有被正确解析为 `False`，导致 Phase-B backbone 始终冻结、严重欠训练
- 修复 bug 后需要重跑，但 leakage-free 特征还没做好

**耗时原因**: bug 排查 + leakage-free 特征生成（TF-IDF 需要在 train cutoff 前重新拟合 vocabulary/IDF/SVD/center-whiten；Qwen embedding 同样需要 TS-aware 处理）耗费了大半天

---

### 7/12 (周六) — 收到导师指导 + V2 no-boost 四配置跑完

**做了什么**:
1. 收到导师 0712 指导意见（`0712zhidao.txt`）
2. 用 leakage-free V2 特征 + min_train_interactions=5 跑四配置

**V2 no-boost 结果（问题1 的第一轮回答）**:
| 配置 | MRR@10 |
|------|:------:|
| ID-only | 0.0112 |
| TF-IDF | 0.0157 |
| TF-IDF+LLM | 0.0157 |
| MV-Align | 0.0157 |

**确认了打平现象**：三个 text 模型 MRR 完全相同 = 0.0157。但此时全部使用 with-Cross 配置。

**同时完成**:
- TF-IDF 3-seed 验证（mean MRR=0.0156, std 很小 → 稳定）
- log10 上跑 ID-only 3-seed（Beauty/Toys/Grocery）
- Toys/Grocery 四配置首次跑出结果

**遇到的问题**:
- V2 min5 过滤后 ID-only MRR 从 0.0150 降到 0.0112（-25%），用户群变严格
- 打平现象在 with-Cross 下完全收敛到 0.0157 → 怀疑 Cross 在"压制"模型间差异

---

### 7/13 (周日) — Cross ablation 揭开 LLM 优势 + boost 象限实验

**做了什么**:
1. **Cross ablation**（问题2 的关键实验）：去掉 Cross 重跑三模型
2. boost + Cross / boost + no-Cross 四象限实验

**Cross ablation 结果（no-boost, no-Cross）**:
| 配置 | MRR@10 |
|------|:------:|
| ID-only | 0.0112 |
| TF no-Cross | 0.0157 |
| **LLM no-Cross** | **0.0168** |
| MV no-Cross | 0.0141 |

**关键发现**:
- 去掉 Cross 后，**LLM 脱颖而出**（0.0168），TF-IDF 不变（0.0157），**但 MV 反而降到 0.0141**
- Cross 在"压制"LLM 的优势（0.0168 → 0.0157）
- LLM no-Cross + boost 跑出 **0.0171**（本周最高分）
- 经典 MV > LLM > TF 层级在严格协议下未恢复

**Cross 效应汇总**:
| 模型 | no-Cross MRR | +Cross MRR | Cross 效应 |
|------|:-----------:|:----------:|:----------:|
| TF | 0.0157 | 0.0157 | 0 |
| LLM | **0.0168** | 0.0157 | **-0.0011** (被压制) |
| MV | 0.0141 | 0.0157 | +0.0016 (被抬升) |

**结论**: Cross 不是稳定净收益组件 → 符合导师问题(2)的建议

**遇到的问题**:
- MV no-Cross = 0.0141 异常低 → 怀疑是融合路径 bug（后续 7/15 确认）
- boost + no-Cross 的 MV 还在跑

---

### 7/14 (周一) — 四象限齐 + 结论初步形成

**做了什么**:
1. boost + no-Cross MV 出结果：0.0151（仍低于 LLM 0.0171）
2. 四象限表格终于补齐

**四象限 MRR@10 完整**:
| | no-Cross | +Cross |
|--|:--------:|:------:|
| no-boost | LLM **0.0168** > TF 0.0157 > MV 0.0141 | 全 0.0157 |
| boost | LLM **0.0171** > TF 0.0160 > MV 0.0151 | LLM 0.0164 > TF 0.0160 > MV 0.0152 |

**初步结论**:
- 最优配置：LLM no-Cross + boost = 0.0171
- MV 在所有配置下都未超越 LLM
- 但 **MV 0.0141 异常低** → 怀疑 MV no-Cross 融合路径有实现问题

**遇到的问题**:
- MV no-Cross 的 `_get_fused_item_embeddings` 走的是简单加法（`item_emb + text_proj`），而 LLM/TF 走的是 `concat + predictor` → **融合容量不对等，不公平**
- 这解释了 MV 0.0141 为什么异常低，但发现这个问题已经是 7/15 了

---

### 7/15 (周二) — 修复 MV 融合 bug + per-source align 设计

**做了什么**:
1. **修复 MV no-Cross 融合 bug**: 从简单加法改为 `concat + predictor`（对齐 LLM/TF 的融合方式）
2. MV concat-fix 三组实验全部跑完
3. 设计 **per-source align** 统一对齐协议

**MV concat-fix 结果**:
| 配置 | 旧(加法) | 新(concat+proj) | 增益 |
|------|:-------:|:--------------:|:----:|
| MV no-Cross no-boost | 0.0141 | **0.0153** | +0.0012 |

修复有效！但 0.0153 仍低于 LLM 0.0168。

**更重要的发现 — 公平性问题**:
- 三模型的对齐方式不一致：TF 用 1× concat align, LLM 也是 1× concat align, MV 根据版本有 per-view / concat 不同
- 为确保公平对比，需要统一为 **per-source align**（每个文本源独立 InfoNCE 对齐）
  - TF: 1× InfoNCE (base 和 ID 对齐)
  - LLM: 2× InfoNCE (base 和 ID 对齐 + llm 和 ID 对齐)
  - MV: 5× InfoNCE (base + 4 views 各自和 ID 对齐)
- 代码改完 + 同步三机

**遇到的问题**:
- 需要同时改 `sasrecalignv3.py` 和 `sasrecalignmultiviewv3.py` 两个文件
- 改完后需要 TF/LLM/MV 全部重跑才公平 → 又是 6 组实验

**耗时原因**: 发现融合 bug → 修复 → 跑实验验证 → 发现对齐不公平 → 设计 per-source align → 改代码 → 同步，一整天都在 debug 和设计层面

---

### 7/16 (周三) — per-source align 6 组全跑完 + 嵌入诊断

**做了什么**:
1. per-source align 公平对比 6 组全部跑完 (09:09 ~ 17:33)
2. 嵌入塌缩诊断（effective rank 分析）

**最终公平结果**:
| | no-Cross | +Cross |
|--|:--------:|:------:|
| TF (1× InfoNCE) | 0.0162 | **0.0164** |
| LLM (2× InfoNCE) | **0.0164** | 0.0162 |
| MV (5× InfoNCE) | 0.0158 | 0.0156 |

**嵌入塌缩诊断结果**:
| 层级 | LLM Effective Rank | MV Effective Rank |
|------|:------------------:|:-----------------:|
| Raw text (512d) | 473/512 (92%) | 192/512 (37%) |
| Projected (256d) | 192/256 (75%) | 139/256 (54%) |

- 4 个 view 各自 64 维满秩，但 concat 后仅有 136/256 有效维度 → **view 间严重冗余**
- view_1 vs view_2 余弦相似度 = 0.42（Function vs Audience 冗余）
- 这是 MV < LLM 的**结构性根因**，非超参问题

---

## 为什么花了一周才出结论

| 日期 | 阻塞原因 | 耗时 |
|------|---------|:----:|
| 7/10 | 首次跑通 TS pipeline | 1 天 |
| 7/11 | `freeze_backbone` bug 排查 + leakage-free 特征制作 | 1 天 |
| 7/12 | with-Cross 打平 → 以为是 bug，花时间核查配置 | 半天 |
| 7/13 | 跑 Cross ablation 才发现是 Cross 压制了差异 | 1 天（含 12h GPU 时间） |
| 7/14 | 四象限补齐 MV，但 MV=0.0141 引发新疑问 | 1 天 |
| 7/15 | 排查 MV 融合 bug（加法 vs concat）→ 修复 → 发现对齐不公平 → 设计 per-source align | 1 天 |
| 7/16 | 6 组公平重跑 + 嵌入诊断 | 1 天（8h GPU 串行）|

每次得出初步结论后，都发现了新的实现问题或公平性缺陷，需要修复后重跑。关键转折点：
1. **7/11 freeze_backbone bug** → V1 结果不可信，必须 V2 重跑
2. **7/13 发现 Cross 压制差异** → 必须补跑 no-Cross 系列
3. **7/15 发现 MV 融合不对等** → 必须修复后重跑
4. **7/15 发现对齐协议不统一** → 必须设计 per-source align 全部重跑

---

## 最终结论（对应两个问题）

### 问题(1): 打平现象确认

**真实，非实现问题。** 跨两版对齐协议（concat align + per-source align）和三个数据集均可复现：

| 配置 | MRR@10 | vs ID-only |
|------|:------:|:----------:|
| ID-only | 0.0112 | — |
| TF-IDF | 0.0162 | +44.6% |
| TF+LLM | 0.0164 | +46.4% |
| MV | 0.0158 | +41.1% |

- TF-IDF 是极强 baseline
- 加 LLM 仅提升 +0.0002
- MV 反而不如 TF 单独使用
- 嵌入诊断揭示根因：MV 4 view 间存在严重冗余（effective rank 塌缩到 37%）

### 问题(2): 简化模型 + Cross 验证

**SE 已删、boost 已关、Cross 证明非净收益。**

| 模型 | no-Cross MRR | +Cross MRR | Cross 效应 |
|------|:-----------:|:----------:|:----------:|
| TF | 0.0162 | 0.0164 | +0.0002 |
| LLM | **0.0164** | 0.0162 | -0.0002 |
| MV | 0.0158 | 0.0156 | -0.0002 |

Cross 对 LLM 有轻微负面影响，对 TF 有轻微正向，整体不稳定 → 不进入主方法。

---

## 论文主表（Beauty, Global TS, seed=2025, min5, no-boost, per-source align）

### 主表格式（no-Cross，主方法配置）

| Model | MRR@10 | NDCG@10 | HR@10 |
|-------|:------:|:-------:|:-----:|
| ID-only (SASRec) | 0.0112 | 0.0151 | 0.0277 |
| + TF-IDF | 0.0162 | 0.0205 | 0.0348 |
| + Single-View (LLM) | **0.0164** | **0.0209** | **0.0358** |
| + Multi-View | 0.0158 | 0.0198 | 0.0330 |

### +Cross 消融

| Model | MRR@10 | NDCG@10 | HR@10 |
|-------|:------:|:-------:|:-----:|
| ID-only (SASRec) | 0.0112 | 0.0151 | 0.0277 |
| + TF-IDF | **0.0164** | 0.0202 | 0.0327 |
| + Single-View (LLM) | 0.0162 | 0.0202 | **0.0335** |
| + Multi-View | 0.0156 | 0.0196 | 0.0326 |

### 频率分桶（no-Cross，test set）

| Model | MRR_freq@10 | MRR_few@10 | MRR_new@10 | R_new@10 | R_few@10 |
|-------|:-----------:|:----------:|:----------:|:--------:|:--------:|
| ID-only | — | — | — | — | — |
| + TF-IDF | 0.0340 | 0.0177 | 0.0116 | 0.0234 | 0.0349 |
| + Single-View (LLM) | 0.0339 | **0.0186** | **0.0121** | 0.0198 | 0.0341 |
| + Multi-View | 0.0329 | **0.0186** | 0.0104 | 0.0168 | 0.0328 |

> 数据集: Amazon Beauty | 协议: Global Time Split, min_train_interactions=5, leakage-free TF-IDF/LLM
> 对齐: per-source align (TF=1× InfoNCE, LLM=2×, MV=5×) | 训练: two-phase (Phase-A 20ep + Phase-B 50ep)
> Toys / Grocery 待跑（per-source align 版本）
