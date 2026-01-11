# CHANGELOG 2025-01-12

## 实验设计与预期结果

### 背景问题

基于 `0111.csv` 的实验结果，识别出以下核心问题：

| 问题 | 数据集 | 现状 | 目标 |
|------|--------|------|------|
| **层级反转** | Toys | TF-IDF+LLM (0.0371) < TF-IDF (0.0373) | LLM > TF-IDF |
| **Scale Law 失效** | Beauty | 7B ≈ 14B ≈ 32B (无差异) | 7B < 14B < 32B |
| **统一参数** | Both | 各自最优参数不同 | 同一组参数通用 |

---

## 新增实验 (7组)

### 1. 统一参数验证

| 实验 | GPU | 配置 | 目的 |
|------|-----|------|------|
| `exp_unified_aggressive_toys` | 5090 | cold=2.5, infer=1.5 | Toys 使用 Beauty 最佳参数 |
| `exp_unified_standard_beauty` | 4090-7 | cold=2.0, infer=1.0 | Beauty 使用 Toys 最佳参数 |

**理论依据**：
- 如果两个数据集能用同一套参数达到接近最优，说明参数具有通用性
- 优先选择 `standard (cold=2.0, infer=1.0)` 作为统一推荐配置

### 2. Toys TF-IDF+LLM 层级修复

| 实验 | GPU | 配置 | 假设 |
|------|-----|------|------|
| `exp_tfidf_llm_toys_align15` | 4090-0 | align=0.15 | LLM 需要更强对齐 |
| `exp_tfidf_llm_toys_tau03` | 4090-1 | tau=0.03 | LLM 需要更尖锐的对比分布 |

**理论依据**：
- Toys 数据集更大 (167k items vs Beauty 67k items)
- 更多样的商品可能需要更强的语义对齐信号
- 低温度 = 更尖锐的 softmax = 更强的正负样本区分

### 3. Beauty Scale Law 验证

| 实验 | GPU | 配置 | 当前 MRR | 预期 |
|------|-----|------|----------|------|
| `exp_boost_14b_aggressive` | 4090-2 | cold=2.5, infer=1.5 | 0.0323 | > 0.0327 |
| `exp_boost_32b_aggressive` | 4090-3 | cold=2.5, infer=1.5 | 0.0325 | > 0.0327 |

**理论依据**：
- 更大模型有更丰富的语义信息，但可能需要更强的 inference boost 才能释放
- 如果 14B/32B + aggressive > 7B + aggressive，说明 Scale Law 被之前的参数掩盖

### 4. Threshold 对比

| 实验 | GPU | 配置 | 覆盖范围 |
|------|-----|------|----------|
| `exp_boost_toys_threshold5` | 4090-4 | threshold=5 | new + 少量 few |

**已有对比**：
- threshold=3: 精准 new [0,3)
- threshold=5: new + 部分 few [0,5)
- threshold=10: new + few [0,10)

---

## 执行命令

```bash
# 5090
GPU_ID=0 nohup bash experiments/exp_unified_aggressive_toys.sh > unified_aggressive_toys.log 2>&1 &

# 4090-0
GPU_ID=0 nohup bash experiments/exp_tfidf_llm_toys_align15.sh > tfidf_llm_toys_align15.log 2>&1 &

# 4090-1
GPU_ID=1 nohup bash experiments/exp_tfidf_llm_toys_tau03.sh > tfidf_llm_toys_tau03.log 2>&1 &

# 4090-2
GPU_ID=2 nohup bash experiments/exp_boost_14b_aggressive.sh > boost_14b_aggressive.log 2>&1 &

# 4090-3
GPU_ID=3 nohup bash experiments/exp_boost_32b_aggressive.sh > boost_32b_aggressive.log 2>&1 &

# 4090-4
GPU_ID=4 nohup bash experiments/exp_boost_toys_threshold5.sh > boost_toys_threshold5.log 2>&1 &

# 4090-7
GPU_ID=7 nohup bash experiments/exp_unified_standard_beauty.sh > unified_standard_beauty.log 2>&1 &
```

---

## 决策树

```
1. 统一参数决策
   如果 unified_aggressive_toys MRR ≥ 0.0376
     且 unified_standard_beauty MRR ≥ 0.0320
       → 使用 standard (cold=2.0, infer=1.0) 作为统一配置
     否则
       → 使用 aggressive (cold=2.5, infer=1.5)，注明 Beauty 最优

2. 层级修复决策
   如果 tfidf_llm_toys_align15 或 tau03 MRR > 0.0373
       → Toys TF-IDF+LLM 需要特殊参数
       → 论文中说明数据集差异
   否则
       → LLM 在 Toys 上确实不如 TF-IDF
       → 需要深入分析 TF-IDF vs LLM 的特性差异

3. Scale Law 决策
   如果 14b_aggressive > 7b_aggressive (0.0327)
     或 32b_aggressive > 7b_aggressive
       → Scale Law 在 aggressive 配置下成立
       → 论文结论：Scale Law 需要合适的推理时参数才能显现
   否则
       → Scale Law 在 Beauty 上不成立
       → 论文结论：数据集规模是 Scale Law 的瓶颈
```

---

## 新增文件清单

### YAML 配置
- `sasrec_align_multi_view_v2_toys_unified_aggressive.yaml`
- `sasrec_align_qwen3_toys_align15.yaml`
- `sasrec_align_qwen3_toys_tau03.yaml`
- `sasrec_align_multi_view_v2_inference_boost_14b_aggressive.yaml`
- `sasrec_align_multi_view_v2_inference_boost_32b_aggressive.yaml`
- `sasrec_align_multi_view_v2_toys_inference_boost_threshold5.yaml`
- `sasrec_align_multi_view_v2_beauty_unified_standard.yaml`

### Shell 脚本
- `experiments/exp_unified_aggressive_toys.sh`
- `experiments/exp_tfidf_llm_toys_align15.sh`
- `experiments/exp_tfidf_llm_toys_tau03.sh`
- `experiments/exp_boost_14b_aggressive.sh`
- `experiments/exp_boost_32b_aggressive.sh`
- `experiments/exp_boost_toys_threshold5.sh`
- `experiments/exp_unified_standard_beauty.sh`

---

## 待验证实验状态

| 实验 | GPU | 状态 |
|------|-----|------|
| exp_inference_boost_toys_14b | 4090-5 | 运行中 |
| exp_inference_boost_toys_32b | 4090-6 | 运行中 |

---

# 公平对比分析 (Fair Comparison)

## 问题识别

### 问题 1：数据集规模相近

| 数据集 | Items | Users | Interactions |
|--------|-------|-------|--------------|
| Beauty | ~67k | ~22k | ~200k |
| Toys | ~167k | ~19k | ~170k |

**结论**：数据规模相近，Scale Law 失效更可能是**参数未优化**而非数据瓶颈。

### 问题 2：参数不公平

当前对比存在的问题：

| 模型 | cold_boost | infer_boost | 问题 |
|------|------------|-------------|------|
| TF-IDF | 0 | 0 | 基准配置 |
| TF-IDF+LLM | 0 | 0 | 基准配置 |
| **Multi-view (aggressive)** | **2.5** | **1.5** | ⚠️ 不公平 |

**核心问题**：Multi-view 的提升可能部分来自 `cold_boost` 和 `infer_boost` 参数，而非模型结构本身。

---

## 参数支持情况

| 参数 | SASRecAlign (TF-IDF/LLM) | MultiViewV2 | 性质 |
|------|--------------------------|-------------|------|
| `cold_start_align_boost` | ✅ 支持 | ✅ 支持 | **共有参数** |
| `inference_cold_text_boost` | ❌ 不支持 | ✅ 支持 | **Multi-view 特有** |

---

## 公平对比实验矩阵

### 贡献分解

| 对比组 | 控制变量 | 证明的贡献 |
|--------|----------|----------|
| TF-IDF vs TF-IDF+LLM (同参数) | 模型结构 | LLM 嵌入的价值 |
| TF-IDF+LLM vs Multi-view (同参数) | 模型结构 | 多视角架构的价值 |
| Multi-view (cold=2, infer=0) vs (cold=2, infer=1) | infer_boost | CHANGE-9 的价值 |

### Beauty 公平对比矩阵

| 模型 | cold_boost | infer_boost | 实验 | 状态 |
|------|------------|-------------|------|------|
| TF-IDF (cold=0) | 0 | 0 | 已有 | ✅ |
| **TF-IDF (cold=2)** | 2.0 | 0 | exp_fair_tfidf_cold2_beauty | ⏳ |
| TF-IDF+LLM (cold=0) | 0 | 0 | 已有 | ✅ |
| **TF-IDF+LLM (cold=2)** | 2.0 | 0 | exp_fair_tfidf_llm_cold2_beauty | ⏳ |
| **Multi-view (cold=0, infer=0)** | 0 | 0 | exp_fair_multiview_no_boost_beauty | ⏳ |
| **Multi-view (cold=2, infer=0)** | 2.0 | 0 | exp_fair_multiview_cold2_only_beauty | ⏳ |
| Multi-view (cold=2, infer=1) | 2.0 | 1.0 | 已有 | ✅ |

### Toys 公平对比矩阵

| 模型 | cold_boost | infer_boost | 实验 | 状态 |
|------|------------|-------------|------|------|
| TF-IDF (cold=0) | 0 | 0 | 已有 | ✅ |
| **TF-IDF (cold=2)** | 2.0 | 0 | exp_fair_tfidf_cold2_toys | ⏳ |
| TF-IDF+LLM (cold=0) | 0 | 0 | 已有 | ✅ |
| **TF-IDF+LLM (cold=2)** | 2.0 | 0 | exp_fair_tfidf_llm_cold2_toys | ⏳ |
| **Multi-view (cold=0, infer=0)** | 0 | 0 | exp_fair_multiview_no_boost_toys | ⏳ |
| **Multi-view (cold=2, infer=0)** | 2.0 | 0 | exp_fair_multiview_cold2_only_toys | ⏳ |
| Multi-view (cold=2, infer=1) | 2.0 | 1.0 | 已有 | ✅ |

---

## 公平对比实验执行命令

### Beauty

```bash
# TF-IDF + cold_boost=2.0
GPU_ID=X nohup bash experiments/exp_fair_tfidf_cold2_beauty.sh > fair_tfidf_cold2_beauty.log 2>&1 &

# TF-IDF+LLM + cold_boost=2.0
GPU_ID=X nohup bash experiments/exp_fair_tfidf_llm_cold2_beauty.sh > fair_tfidf_llm_cold2_beauty.log 2>&1 &

# Multi-view 无 boost
GPU_ID=X nohup bash experiments/exp_fair_multiview_no_boost_beauty.sh > fair_mv_no_boost_beauty.log 2>&1 &

# Multi-view cold=2.0 only (无 infer_boost)
GPU_ID=X nohup bash experiments/exp_fair_multiview_cold2_only_beauty.sh > fair_mv_cold2_only_beauty.log 2>&1 &
```

### Toys

```bash
# TF-IDF + cold_boost=2.0
GPU_ID=X nohup bash experiments/exp_fair_tfidf_cold2_toys.sh > fair_tfidf_cold2_toys.log 2>&1 &

# TF-IDF+LLM + cold_boost=2.0
GPU_ID=X nohup bash experiments/exp_fair_tfidf_llm_cold2_toys.sh > fair_tfidf_llm_cold2_toys.log 2>&1 &

# Multi-view 无 boost
GPU_ID=X nohup bash experiments/exp_fair_multiview_no_boost_toys.sh > fair_mv_no_boost_toys.log 2>&1 &

# Multi-view cold=2.0 only (无 infer_boost)
GPU_ID=X nohup bash experiments/exp_fair_multiview_cold2_only_toys.sh > fair_mv_cold2_only_toys.log 2>&1 &
```

---

## 预期结果与分析框架

### 1. 多视角架构贡献 (Architecture Contribution)

```
对比: Multi-view (cold=X, infer=0) vs TF-IDF+LLM (cold=X)
预期: Multi-view > TF-IDF+LLM (因为多视角结构)
如果相等: 多视角结构无额外贡献，提升全来自 CHANGE-9
```

### 2. LLM 嵌入贡献 (LLM Embedding Contribution)

```
对比: TF-IDF+LLM (cold=X) vs TF-IDF (cold=X)
预期: TF-IDF+LLM > TF-IDF (因为 LLM 语义)
如果相等或反转: LLM 嵌入在该配置下无额外贡献
```

### 3. CHANGE-9 贡献 (Inference Boost Contribution)

```
对比: Multi-view (cold=2, infer=1) vs Multi-view (cold=2, infer=0)
预期: 前者 HR_new 更高 (因为推理时冷启动加权)
这是 Multi-view 特有能力，公平比较下的真正创新
```

### 4. Cold Boost 贡献 (Training-time Cold Boost)

```
对比: 任意模型 (cold=2) vs 同模型 (cold=0)
预期: cold=2 的低频指标更好
这是共有参数，不是 Multi-view 的独特贡献
```

---

## 论文叙述建议

基于公平对比，论文应该分层叙述贡献：

1. **基础贡献**：TF-IDF 特征 + 对齐损失带来的基础提升
2. **LLM 贡献**：LLM 嵌入相比 TF-IDF 的边际提升（可能很小）
3. **多视角架构贡献**：Multi-view 相比单视角的提升（需公平对比确认）
4. **CHANGE-9 贡献**：inference_cold_text_boost 带来的 HR_new 提升（Multi-view 特有）

---

## 新增公平对比脚本清单

### Beauty
- `experiments/exp_fair_tfidf_cold2_beauty.sh`
- `experiments/exp_fair_tfidf_llm_cold2_beauty.sh`
- `experiments/exp_fair_multiview_no_boost_beauty.sh`
- `experiments/exp_fair_multiview_cold2_only_beauty.sh`

### Toys
- `experiments/exp_fair_tfidf_cold2_toys.sh`
- `experiments/exp_fair_tfidf_llm_cold2_toys.sh`
- `experiments/exp_fair_multiview_no_boost_toys.sh`
- `experiments/exp_fair_multiview_cold2_only_toys.sh`
