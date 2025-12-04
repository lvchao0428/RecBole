# 白化特征效果变差分析报告

## 📊 实验结果对比

| 配置 | Recall@10 | MRR@10 | NDCG@10 | 相对变化 |
|------|-----------|--------|---------|----------|
| **Baseline (1203)** | 0.0677 | 0.0238 | 0.0342 | - |
| **Whiten + No Early Jump** | 0.0511 | 0.0285 | 0.0338 | R↓24.5%, M↑19.7% |

## 🔍 核心问题诊断

### 1. **特征尺度问题**（主要原因）

**现象**：从验证日志看到
```
ℹ️  Embedding未L2归一化 (mean=34.3805, std=5.2389)
```

**原因分析**：
- **原始TF-IDF特征**: L2归一化，norm ≈ 1
- **白化后特征**: 协方差矩阵 = I，但 norm >> 1 (mean=34.38)

**数学解释**：
白化变换 `X_white = (X - μ) @ W` 使得 `Cov(X_white) = I`，但这**不保证** `||X_white|| = 1`

对于InfoNCE loss:
```python
sim = cosine_similarity(seq, text) / temperature
    = dot(seq, text) / (||seq|| * ||text|| * tau)
```

- 如果 `||text|| ≈ 34` 而不是1，cosine相似度的数值范围改变
- `temperature=0.07` 对于大norm的特征可能太小，导致 `sim/tau` 过大
- Softmax饱和 → 梯度消失 → 训练不稳定

### 2. **对比学习难度改变**

白化后：
- 特征维度间相关性被消除（好处）
- 所有维度方差相等 → 信息密度降低
- 正负样本区分度可能改变

**表现**：
- MRR上升（排序质量提高）← 可能是模型变"保守"，对top-1更自信
- Recall下降（召回范围缩小）← 多样性降低，漏掉了一些相关物品

### 3. **Phase-A过拟合**

删除"early jump"意味着：
- Phase-A跑满8 epochs而不提前终止
- Text projection head可能过度拟合对齐任务
- 损失了embedding的泛化能力

## 🔧 修复方案

### 方案1: 白化后L2归一化（推荐★★★★★）

**原理**：恢复特征norm ≈ 1，同时保持去相关性

**实现**：
```bash
# 1. 对已生成的白化embedding做L2归一化
python tools/normalize_whitened_emb.py \
    --input dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
    --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy

# 2. 使用归一化后的embedding训练
# 修改配置文件中的路径：
# item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy
```

**或者在生成时添加**：修改 `build_item_text_emb_qwen3_hf.py` 的 `_center_whiten_and_normalize` 函数，在白化后添加：
```python
# After whitening
emb_whitened = emb_centered @ whiten_matrix
emb_whitened[0, :] = 0.0

# ADD THIS: L2 normalize to restore norm ≈ 1
norms = np.linalg.norm(emb_whitened[1:], axis=1, keepdims=True)
emb_whitened[1:] = emb_whitened[1:] / np.clip(norms, 1e-8, None)

return emb_whitened.astype(np.float32)
```

**预期效果**：
- 特征norm恢复到1
- Temperature可以继续使用0.07左右的值
- 对比学习更稳定

---

### 方案2: 调整Temperature和Alignment Weight（快速测试）

**配置**：使用 `sasrec_align_base_whiten.yaml`

**关键参数调整**：
```yaml
temperature: 0.15              # 0.07 → 0.15 (适应大norm特征)
alignment_weight: 0.05         # 0.1 → 0.05 (降低对齐权重)
cross_dropout_prob: 0.3        # 0.0 → 0.3 (防止过拟合)
normalize_text: false          # 白化后不再L2归一化
```

**训练命令**：
```bash
bash two_phase_whiten_test.sh
```

该脚本会：
- Grid search: `temperature ∈ {0.1, 0.15, 0.2}`
- Grid search: `alignment_weight ∈ {0.03, 0.05, 0.08}`
- Phase-A只跑5 epochs（避免过拟合）
- 启用early stopping with baseline gate

**预期效果**：
- 更大的temperature缓解softmax饱和
- 更小的alignment_weight减少对文本的依赖
- Shorter Phase-A避免过拟合

---

### 方案3: 恢复Phase-A Early Stopping（防止过拟合）

**原理**：Phase-A的目标是warm-up text head，不需要跑太久

**修改 `two_tfidf_stage.sh`**：
```bash
python scripts/exp_align_two_stage.py \
  --dataset Amazon_Beauty \
  --mode base \
  --stage1_epochs 5 \              # 8 → 5 (减少epochs)
  --stage2_epochs 50 \
  --stage2_lr 3e-5 \
  --temperatures 0.1 0.15 \        # 增大temperature
  --weights 0.05 \                 # 减小weight
  --exclude_topk 0 \
  --base_emb dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy \  # 使用归一化的
  --ndcg_baseline 0.0285 \         # 添加baseline gate
  --ndcg_gain_threshold 0.02       # 要求提升2%才进入Phase-B
```

**预期效果**：
- Phase-A不会过度优化
- 只有达标的配置才进入Phase-B
- 减少计算浪费

---

## 🎯 推荐执行顺序

### Step 1: L2归一化白化后的embedding（必做）

```bash
python tools/normalize_whitened_emb.py \
    --input dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
    --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy
```

### Step 2: 验证归一化效果

```bash
python tools/verify_whiten.py \
    dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy \
    --dataset Amazon_Beauty
```

预期看到：
```
✅ Embedding已L2归一化 (mean≈1.0, std≈0.0)
✅ Embedding已中心化 (mean_abs≈0.0)
✅ 协方差矩阵对角线≈1.0
```

### Step 3: 使用归一化的embedding + 调整超参数训练

```bash
# 修改 two_tfidf_stage.sh 中的 --base_emb 路径
# 然后运行
bash two_phase_whiten_test.sh
```

### Step 4: 对比结果

预期改进：
- Recall@10 恢复到 0.065+ (接近baseline)
- MRR@10 保持在 0.028+ (保持提升)
- NDCG@10 提升到 0.036+ (综合提升)

---

## 📌 关键要点总结

1. **白化≠归一化**：白化使协方差=I，但不保证norm=1
2. **Temperature依赖特征尺度**：大norm特征需要大temperature
3. **Phase-A要适度**：warm-up不是fine-tune，跑太久会过拟合
4. **监控指标平衡**：MRR↑但Recall↓说明模型变保守了

---

## 🔬 补充实验建议

1. **消融实验**：分别测试
   - 只L2归一化，不调超参
   - 只调超参，不归一化
   - 两者结合

2. **Temperature敏感性分析**：
   - 测试 `tau ∈ {0.05, 0.07, 0.1, 0.15, 0.2}`
   - 绘制 Recall/MRR vs Temperature曲线

3. **白化vs不白化对比**：
   - 相同超参下，对比原始TF-IDF vs Whitened (L2-normed)
   - 理论上白化后应该更好（去相关性）

---

## 📝 代码Bug修复记录

### Bug #1: 白化矩阵计算错误（已修复）

**位置**: `_center_whiten_and_normalize` 函数

**错误代码**:
```python
whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
```

**正确代码**:
```python
D_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-5))
whiten_matrix = U @ D_inv_sqrt @ U.T  # 缺少了 @ U.T
```

**影响**: 白化后的协方差矩阵不是单位矩阵，导致验证失败

### Bug #2: enable_whiten变量未定义（已修复）

**问题**: 在使用前未定义
**修复**: 移动定义到使用前

---

## 📚 参考资料

1. **Whitening Transform**: https://en.wikipedia.org/wiki/Whitening_transformation
2. **InfoNCE Loss**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
3. **Temperature in Contrastive Learning**: "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"

