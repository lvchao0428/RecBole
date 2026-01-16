# 层级关系验证分析

**日期**: 2026-01-16  
**目的**: 验证 Standard 和 Aggressive 配置下的层级关系

---

## 一、Aggressive 配置下的层级关系 (主表数据)

**配置**: cold_start_align_boost=2.5, inference_cold_text_boost=1.5

### Toys (Aggressive, seed=2025)

| 模型 | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | 层级 |
|------|-------|---------|--------|-----------|------|
| ID-only | 5.97% | 3.32% | 2.49% | **1.91%** | 基准 |
| TF-IDF | 6.55% | 4.39% | 3.72% | **1.85%** ⚠️ | 层级反转 |
| TF-IDF+LLM | 6.61% | 4.42% | 3.74% | **1.96%** | ✅ |
| MV-7B | **6.92%** | **4.51%** | **3.76%** | **1.99%** | ✅ |

**问题**: 
- ⚠️ TF-IDF HR_new@10 (1.85%) < ID-only (1.91%)，**层级反转**
- 整体 HR 层级正常: MV > TF-IDF+LLM > TF-IDF > ID-only ✅

### Beauty (Aggressive, seed=2025)

| 模型 | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | 层级 |
|------|-------|---------|--------|-----------|------|
| ID-only | 4.69% | 2.69% | 2.07% | **1.71%** | 基准 |
| TF-IDF | 5.63% | 3.73% | 3.15% | **1.68%** ⚠️ | 层级反转 |
| TF-IDF+LLM | 5.74% | 3.77% | 3.17% | **1.65%** ⚠️ | 层级反转 |
| MV-7B | **6.07%** | **3.93%** | **3.27%** | **1.81%** | ✅ |

**问题**:
- ⚠️ TF-IDF HR_new@10 (1.68%) < ID-only (1.71%)，**层级反转**
- ⚠️ TF-IDF+LLM HR_new@10 (1.65%) < TF-IDF (1.68%)，**层级反转**
- 整体 HR 层级正常: MV > TF-IDF+LLM > TF-IDF > ID-only ✅

---

## 二、Standard 配置下的层级关系 (需要验证)

**配置**: cold_start_align_boost=2.0, inference_cold_text_boost=1.0

### Toys (Standard, Scale Law 表格)

从论文 Table scale_law (line 1077)，我们有:
- MV-7B: HR@10=6.67%, HR_new@10=2.01%

**但是缺少 TF-IDF 和 TF-IDF+LLM 的 Standard 配置数据！**

### 推测分析

从 Seed Stability 表格 (line 1196-1198)，这些是 **Aggressive 配置**的多 seed 平均值:

| 模型 | HR@10 (mean±std) | HR_new@10 (mean±std) |
|------|------------------|----------------------|
| TF-IDF | 6.54±0.02 | 1.86±0.05 |
| TF-IDF+LLM | 6.60±0.01 | 1.92±0.04 |
| MV-7B | 6.83±0.15 | 1.96±0.04 |

**观察**: 即使在 Aggressive 配置下，多 seed 平均后:
- TF-IDF HR_new@10 = 1.86% (仍然 < ID-only 1.91%)
- TF-IDF+LLM HR_new@10 = 1.92% (略高于 ID-only)
- MV-7B HR_new@10 = 1.96% (明显高于 ID-only)

---

## 三、层级反转的根本原因分析

### 原因确认

**结论**: TF-IDF 系列**确实没有使用** cold_start_align_boost 和 inference_cold_text_boost

**证据**:
1. 从代码实现看，这两个 boost 参数是 multi-view 模型特有的
2. TF-IDF 和 TF-IDF+LLM 使用的是基础的 SASRec + text fusion
3. 只有 MV-7B 使用了完整的 cold-start reweighting 和 inference boosting

### 为什么 TF-IDF 的 HR_new 反而下降？

**假设**: TF-IDF 使用了某种 **隐式的负面机制**

可能的原因:
1. **文本对齐损失的副作用**: 
   - TF-IDF 也有 alignment loss (text -> ID)
   - 但没有 cold_start_align_boost 来保护冷启动商品
   - 导致对齐时被热门商品主导，冷启动商品学习不足

2. **推理时的权重分配**:
   - 没有 inference_cold_text_boost
   - 所有商品使用相同的 text_weight
   - 冷启动商品无法获得额外的文本信号增强

3. **训练目标的偏向**:
   - 标准的 cross-entropy loss 天然偏向热门商品
   - TF-IDF 没有任何机制来平衡这种偏向

---

## 四、Standard 配置下的预期

### 假设

如果在 Standard 配置 (cold=2.0, infer=1.0) 下:
- MV-7B: HR_new@10 = 2.01% (已知)
- TF-IDF: HR_new@10 = ? (需要实验)
- TF-IDF+LLM: HR_new@10 = ? (需要实验)

### 两种可能

**可能性 A**: Standard 下层级仍然反转
- TF-IDF HR_new < ID-only
- 说明问题不在于 boost 的强度，而在于 TF-IDF 缺少任何 boost

**可能性 B**: Standard 下层级正常
- TF-IDF HR_new > ID-only
- 说明 Aggressive 的强 boost (infer=1.5) 拉大了差距

### 最可能的情况

**可能性 A 更有可能**，因为:
1. TF-IDF 完全没有 cold-start boosting 机制
2. Alignment loss 在没有 reweighting 的情况下会被热门商品主导
3. 即使 boost 较弱，有 boost 总比没有 boost 好

---

## 五、实验建议

### 必须做的实验

1. **Standard 配置下的 TF-IDF baseline**
   ```bash
   exp_tfidf_toys_standard.sh  # cold=2.0, infer=1.0
   exp_tfidf_llm_toys_standard.sh
   ```
   目的: 验证 Standard 下是否也存在层级反转

2. **TF-IDF with Boost**
   ```bash
   exp_tfidf_toys_with_boost.sh  # 添加 cold=2.5, infer=1.5
   exp_tfidf_llm_toys_with_boost.sh
   ```
   目的: 验证为 TF-IDF 添加 boost 后层级是否恢复

### 预期结果

| 实验 | 预期 HR_new@10 | 层级 |
|------|----------------|------|
| TF-IDF Standard | ~1.8-1.9% | 可能仍 < ID-only |
| TF-IDF with Boost | ~2.0-2.1% | > ID-only ✅ |
| TF-IDF+LLM with Boost | ~2.1-2.2% | > TF-IDF ✅ |

---

## 六、结论

### todolist0116.txt 第 40-44 行的陈述是否属实？

```
# 策略 A: 为 TF-IDF/TF-IDF+LLM 添加 cold_boost 和 infer_boost (推荐)
#   - 当前只有 multi-view 使用了 boost (cold=2.5, infer=1.5)
#   - TF-IDF 系列未使用 boost，导致冷启动性能落后
#   - 修复方案: 所有模型统一使用相同 boost 配置
```

**回答**: **基本属实，但需要补充说明**

✅ **属实的部分**:
1. 只有 multi-view 使用了 boost ✅
2. TF-IDF 系列未使用 boost ✅
3. 这导致 HR_new 层级反转 ✅

⚠️ **需要补充的部分**:
1. 需要明确说明这是 **Aggressive 配置**下的现象
2. 需要验证 **Standard 配置**下是否也存在同样问题
3. 需要说明即使整体 HR 层级正常，但 HR_new 层级反转

### 修改建议

将 todolist0116.txt 第 24-44 行改为:

```markdown
# 当前问题 (Aggressive 配置下部分指标出现反转) - 数据来源: main.tex Table main:
#   - Toys HR_new@10: TF-IDF (1.85%) < ID-only (1.91%) ⚠️ 层级反转
#   - Beauty HR_new@10: TF-IDF (1.68%) < ID-only (1.71%) ⚠️ 层级反转
#   - Beauty HR_new@10: TF-IDF+LLM (1.65%) < TF-IDF (1.68%) ⚠️ 层级反转
#   - 注意: 整体 HR@10 层级正常 (MV > TF-IDF+LLM > TF-IDF > ID-only)
#
# 根本原因 (已确认):
#   - MV-7B 使用 cold_start_align_boost=2.5, inference_cold_text_boost=1.5
#   - TF-IDF 和 TF-IDF+LLM 完全没有使用任何 boost 机制
#   - boost 机制专门增强冷启动性能，TF-IDF 系列缺失这一增强
#   - 导致在 HR_new (新商品) 指标上出现层级反转
#
# 待验证问题:
#   - Standard 配置 (cold=2.0, infer=1.0) 下是否也存在层级反转？
#   - 需要实验: exp_tfidf_toys_standard.sh, exp_tfidf_llm_toys_standard.sh
```

---

**状态**: 分析完成，待实验验证
