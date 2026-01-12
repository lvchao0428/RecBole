# 对话上下文摘要 (2025-01-12)

> 用于在其他设备快速恢复上下文继续讨论

---

## 🎯 核心目标

1. **层级验证**：`multi-view > tfidf+llm > tfidf > 50ep(id only)`
2. **Scale Law 验证**：7B < 14B < 32B（整体或特定指标）
3. **统一参数**：Beauty 和 Toys 用同一组参数达标

---

## 📊 当前最佳结果 (0111.csv)

### Beauty

| 模型 | MRR@10 | NDCG@10 | HR@10 | 层级 |
|------|--------|---------|-------|------|
| base50ep | 0.0207 | 0.0269 | 0.0469 | 基准 |
| TF-IDF | 0.0318 | 0.0377 | 0.0569 | ✅ |
| TF-IDF+LLM | 0.0320 | 0.0381 | 0.0579 | ✅ |
| **MV-7B (aggressive)** | **0.0327** | **0.0393** | **0.0607** | ✅ |

**Beauty 结论**：层级 ✅ 达标，Scale Law ❌ 不成立 (7B ≈ 14B ≈ 32B)

### Toys

| 模型 | MRR@10 | NDCG@10 | HR@10 | 层级 |
|------|--------|---------|-------|------|
| base50ep | 0.0249 | 0.0332 | 0.0597 | 基准 |
| TF-IDF | **0.0373** | 0.0441 | 0.0660 | ✅ |
| TF-IDF+LLM | 0.0371 | 0.0441 | 0.0666 | ⚠️ MRR < TF-IDF |
| **MV-7B (boost_toys)** | 0.0376 | **0.0451** | **0.0695** | ✅ |

**Toys 结论**：层级 ⚠️ TF-IDF+LLM MRR 略低，Scale Law ✅ HR@10/HR_new 成立

---

## 🔧 关键技术变更 (CHANGE-9)

### inference_cold_text_boost

**问题**：Multi-view 模型 MRR 很高，但 HR_new 下降

**解决方案**：推理时给冷启动商品更强的文本权重

```python
# sasrecalignmultiviewv2.py
if self.inference_cold_text_boost > 0 and item_ids is not None:
    item_pop = self.item_popularity[item_ids].float()
    threshold = float(self.cold_start_align_threshold)
    cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold
    cold_boost = 1.0 + self.inference_cold_text_boost * cold_factor
    effective_text_weight = base_effective_text_weight * cold_boost
```

**效果**：
- Beauty: HR_new 从 -2.34% → +5.26%
- 整体 MRR 同时提升

---

## 📁 关键配置参数

### 标准配置 (standard)
```yaml
cold_start_align_boost: 2.0
inference_cold_text_boost: 1.0
cold_start_align_threshold: 10
alignment_weight: 0.10
temperature: 0.05
```

### 激进配置 (aggressive)
```yaml
cold_start_align_boost: 2.5
inference_cold_text_boost: 1.5
cold_start_align_threshold: 10
alignment_weight: 0.10
temperature: 0.05
```

---

## 🏃 运行中的实验

| 实验 | GPU | 数据集 | 目的 |
|------|-----|--------|------|
| exp_inference_boost_toys_14b | 4090-5 | Toys | Scale Law |
| exp_inference_boost_toys_32b | 4090-6 | Toys | Scale Law |
| exp_unified_aggressive_toys | 5090 | Toys | 统一参数 |
| exp_tfidf_llm_toys_align15 | 4090-0 | Toys | 修复层级 |
| exp_tfidf_llm_toys_tau03 | 4090-1 | Toys | 修复层级 |
| exp_boost_14b_aggressive | 4090-2 | Beauty | Scale Law |
| exp_boost_32b_aggressive | 4090-3 | Beauty | Scale Law |
| exp_boost_toys_threshold5 | 4090-4 | Toys | Threshold |
| exp_unified_standard_beauty | 4090-7 | Beauty | 统一参数 |

---

## 📋 待解决问题

### 1. Toys TF-IDF+LLM < TF-IDF
- **假设 A**：LLM 需要更高的 alignment_weight (0.15)
- **假设 B**：LLM 需要更低的 temperature (0.03)
- **实验**：`exp_tfidf_llm_toys_align15`, `exp_tfidf_llm_toys_tau03`

### 2. Beauty Scale Law 不成立
- **假设**：参数未最优化，需要 aggressive 配置
- **实验**：`exp_boost_14b_aggressive`, `exp_boost_32b_aggressive`

### 3. 统一参数
- **候选**：standard (cold=2.0, infer=1.0)
- **验证**：`exp_unified_aggressive_toys`, `exp_unified_standard_beauty`

### 4. ⚠️ 公平对比问题 (新增)
- **问题**：Multi-view 使用 aggressive 参数 (cold=2.5, infer=1.5)，而基准模型用 cold=0
- **影响**：提升可能部分来自参数而非模型结构
- **解决方案**：重跑基准模型使用相同的 cold_boost 参数

#### 参数支持情况
| 参数 | TF-IDF/LLM | Multi-view |
|------|------------|------------|
| cold_start_align_boost | ✅ 支持 | ✅ 支持 |
| inference_cold_text_boost | ❌ 不支持 | ✅ 支持 |

#### 公平对比实验 (8组)
```bash
# Beauty
exp_fair_tfidf_cold2_beauty.sh        # TF-IDF + cold=2.0
exp_fair_tfidf_llm_cold2_beauty.sh    # TF-IDF+LLM + cold=2.0
exp_fair_multiview_no_boost_beauty.sh # Multi-view 无 boost
exp_fair_multiview_cold2_only_beauty.sh # Multi-view cold=2.0 无 infer

# Toys (同上)
exp_fair_tfidf_cold2_toys.sh
exp_fair_tfidf_llm_cold2_toys.sh
exp_fair_multiview_no_boost_toys.sh
exp_fair_multiview_cold2_only_toys.sh
```

#### 贡献分解框架
1. **架构贡献**: MV(cold=X, infer=0) vs LLM(cold=X)
2. **LLM贡献**: LLM(cold=X) vs TF-IDF(cold=X)
3. **CHANGE-9贡献**: MV(cold=2, infer=1) vs MV(cold=2, infer=0) ← Multi-view 特有

---

## 📂 重要文件路径

### 数据
- `paper_sigir/0111.csv` - 最新实验结果

### 配置
- `sasrec_align_multi_view_v2_inference_boost*.yaml` - inference_boost 系列
- `sasrec_align_multi_view_v2_*_unified_*.yaml` - 统一参数验证

### 代码
- `recbole/model/sequential_recommender/sasrecalignmultiviewv2.py` - Multi-view V2 模型
- `scripts/two_phase_train.py` - 两阶段训练脚本

### 日志
- `paper_sigir/CHANGELOG_0112.md` - 本次变更详情
- `paper_sigir/CHANGELOG_0111.md` - CHANGE-9 详情

---

## 🔍 Scale Law 深度分析结论 (0112)

### 已验证发现

- ❌ **Scale Law 在所有物品分组上均不成立**
  - new/few/frequent 分层均无 Scale Law
  - 即使冷启动场景 (new items)，7B 也始终优于或持平 14B/32B
  - 高频物品上 Scale Law 完全反转 (Beauty: 7B > 14B > 32B)

### 核心结论

> "Multi-view 文本特征对高频物品的边际效益有限"  
> "7B 模型已足够捕获推荐任务所需的语义信息"

---

## 💡 新假设：SVD 压缩比导致 Scale Law 失效

### 问题

所有模型的 LLM 嵌入都压缩到同样的 **64D**：

| 模型 | 原始维度 | 压缩后 | 压缩比 |
|------|----------|--------|--------|
| 7B | 3584D | 64D | **56x** |
| 14B | 5120D | 64D | **80x** ⚠️ |
| 32B | 5120D | 64D | **80x** ⚠️ |

**结论**：更大模型信息损失比例更高，抵消了语义优势

### 假设

维度自适应压缩才能公平验证 Scale Law：

| 模型 | 建议维度 | 压缩比 |
|------|----------|--------|
| 7B | 64D | 56x (保持) |
| 14B | 128D | **40x** (降低) |
| 32B | 256D | **20x** (显著降低) |

### 待验证实验

- TODO: `exp_svd_14b_128` - 14B 模型 SVD 到 128 维
- TODO: `exp_svd_32b_256` - 32B 模型 SVD 到 256 维
- TODO: `exp_proj_14b_128to64` - 14B(128D) → 投影到 64D 再融合
- TODO: `exp_attn_32b_256` - 32B(256D) + 注意力融合

**预期**：如果假设成立，exp_svd_32b_256 > exp_svd_14b_128 > exp_svd_7b_64 (Scale Law 恢复!)

---

## 🎯 下一步行动

1. ✅ 等待运行中的 8 个公平对比实验完成
2. 根据结果更新 `0111.csv`
3. 按照决策树判断：
   - 统一参数是否可行
   - 层级是否完全达标
   - 各组件独立贡献分解
4. ⏳ 验证 SVD 压缩比假设（需修改预处理流程）
5. 更新论文结论

---

## 💡 快速恢复对话

```
请读取以下文件了解上下文：
1. paper_sigir/CONTEXT_SUMMARY.md - 对话摘要
2. paper_sigir/CHANGELOG_0112.md - 最新实验设计
3. paper_sigir/0111.csv - 实验数据

然后告诉我最新的实验结果，我会帮你分析并给出下一步建议。
```
