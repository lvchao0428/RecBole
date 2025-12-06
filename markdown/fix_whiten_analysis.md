# Whiten 效果不佳的原因分析与解决方案

## 问题现象

验证日志显示：
```
协方差矩阵分析 (sample_size=500):
   - 对角线均值: 0.8418 (期望≈1.0)  ❌
   - 对角线标准差: 0.1682 (期望≈0)  ❌
   - 非对角线最大值: 0.6651 (期望≈0)  ⚠️
   - 非对角线平均值: 0.0675 (期望≈0)  ✅
```

## 根本原因

### 1. Pre-SVD L2归一化破坏协方差结构

在 `build_item_text_emb_base.py` 第167-168行：
```python
if pre_svd_l2:
    tfidf = l2_normalize(tfidf, norm="l2", axis=1, copy=False)
```

**问题**：
- TF-IDF向量在SVD前被L2归一化到单位球面
- L2归一化后的向量协方差矩阵 ≠ 原始向量的协方差矩阵
- SVD在"扭曲"的协方差空间中进行降维
- 白化基于SVD输出计算，但这些向量的协方差已经不"自然"

### 2. 数学分析

对于L2归一化的向量 $\mathbf{x}'_i = \frac{\mathbf{x}_i}{||\mathbf{x}_i||}$：

- 原始协方差：$\text{Cov}(\mathbf{X}) = \frac{1}{n}\mathbf{X}^T\mathbf{X}$
- L2归一化后：$\text{Cov}(\mathbf{X}') = \frac{1}{n}\sum_i \frac{\mathbf{x}_i \mathbf{x}_i^T}{||\mathbf{x}_i||^2}$ （权重不均）
- SVD降维：$\mathbf{Z} = \text{SVD}(\mathbf{X}')$ （在扭曲空间中降维）
- 白化：$\mathbf{W} = \mathbf{U}\text{diag}(1/\sqrt{S})$ 基于 $\text{Cov}(\mathbf{Z})$

但 $\text{Cov}(\mathbf{Z})$ 的特征值分布已经不反映真实数据的方差结构！

### 3. 为什么对角线≠1.0？

白化矩阵计算基于**训练集**的协方差：
```python
train_centered = train_emb - mean
cov = (train_centered.T @ train_centered) / len(train_centered)  # 训练集协方差
U, S, _ = np.linalg.svd(cov)
whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
```

但验证时计算的是**全部item（或采样）**的协方差：
```python
sample_emb = emb[indices]  # 可能包含test/valid set
cov_after = (sample_emb.T @ sample_emb) / len(sample_emb)
```

**两个原因导致对角线≠1.0**：
1. **样本不同**：训练集 vs 验证时的采样（可能包含test/valid）
2. **Pre-SVD L2归一化**：破坏了自然协方差结构，白化基于错误的方差

## 解决方案

### 方案A：禁用 pre_svd_l2（推荐用于白化）

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --no_pre_svd_l2   # 关键：禁用SVD前的L2归一化
```

**优点**：
- 保留自然协方差结构
- 白化效果更准确（Cov → I）
- 方差信息被正确保留到白化阶段

**缺点**：
- SVD数值稳定性可能略降（但TF-IDF向量通常已经归一化，影响不大）

### 方案B：在SVD后、白化前重新L2归一化

修改 `_fit_tfidf_svd` 和 `_fit_on_train_transform_all`：

```python
# SVD transform
reduced = svd.transform(tfidf_all)

# 重新L2归一化（消除pre-SVD归一化的影响）
reduced = l2_normalize(reduced, norm="l2", axis=1)

# 白化将基于L2归一化后的向量计算协方差
```

**优点**：
- 保持pre-SVD L2归一化（数值稳定）
- 白化基于一致的L2归一化空间

**缺点**：
- 所有向量在单位球面上，方差结构单一
- 白化效果可能仍然不理想（因为Cov特殊）

### 方案C：仅中心化，不白化（最简单）

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --no_whiten   # 仅中心化 + L2归一化
```

**优点**：
- 简单可靠
- 中心化已经消除均值偏差
- L2归一化统一scale

**缺点**：
- 特征间仍可能有相关性（未去相关）

## 推荐方案

### 对于TF-IDF + SVD (base embeddings)：

**使用方案A（禁用pre_svd_l2）+ 白化**

理由：
- TF-IDF本身已经有归一化（norm='l2' in TfidfVectorizer默认值）
- SVD不需要额外的pre-L2归一化
- 保留自然协方差结构，白化效果最佳

### 对于LLM embeddings (Qwen3)：

**保持当前实现（有白化）**

理由：
- LLM embeddings已经在encode时做了L2归一化（encode_batch函数第214行）
- 白化基于L2归一化后的向量，虽然对角线可能≠1.0，但去相关性仍然有效
- 实践中效果良好

## 重新生成建议

### 1. 重新生成 TF-IDF embeddings（推荐）

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.v2.npy \
  --svd_dim 256 \
  --no_pre_svd_l2 \
  --dtype float16

# 验证
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.v2.npy
```

预期结果：
- 对角线均值: 0.95-1.05 ✅
- 非对角线平均值: < 0.05 ✅

### 2. 保持 Qwen3 embeddings 不变

当前实现已经足够好，无需修改。

## 数值容差分析

即使对角线均值=0.84，白化仍然**部分有效**：

| 指标 | 期望值 | 当前值 | 评价 |
|------|--------|--------|------|
| 对角线均值 | 1.0 | 0.8418 | 偏差16% (可接受) |
| 对角线std | 0.0 | 0.1682 | 较大 (需改进) |
| 非对角线均值 | 0.0 | 0.0675 | 良好 |
| 非对角线最大值 | 0.0 | 0.6651 | 存在强相关 |

**结论**：白化减少了相关性（非对角线平均值较小），但未完全去相关（存在0.67的强相关对）。

## 总结

1. **根本原因**：pre-SVD L2归一化破坏了协方差结构
2. **推荐方案**：重新生成TF-IDF embeddings，禁用`--no_pre_svd_l2`
3. **验证标准**：对角线均值0.9-1.1，非对角线平均值<0.05
4. **可接受性**：当前0.84虽然不完美，但白化仍有一定效果（去相关性）

## 下一步行动

```bash
# Step 1: 重新生成 base embeddings
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --no_pre_svd_l2

# Step 2: 验证
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# Step 3: 如果仍有问题，考虑方案C（仅中心化）
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --no_whiten
```

