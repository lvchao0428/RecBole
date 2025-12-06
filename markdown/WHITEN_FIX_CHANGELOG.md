# Whitening效果修复 - Changelog

**日期**: 2025-12-03  
**修复文件**: `tools/build_item_text_emb_base.py`  
**问题类型**: 白化效果不理想（协方差矩阵对角线≠1.0）

---

## 🐛 问题描述

### 症状
验证 TF-IDF embeddings 的白化效果时发现：
```
协方差矩阵分析 (sample_size=500):
   - 对角线均值: 0.8418 (期望≈1.0)  ❌
   - 对角线标准差: 0.1682 (期望≈0)  ❌
   - 非对角线最大值: 0.6651 (期望≈0)  ⚠️
   - 非对角线平均值: 0.0675 (期望≈0)  ✅
```

### 期望结果
白化后的协方差矩阵应接近单位矩阵：
- 对角线元素 ≈ 1.0（方差=1）
- 非对角线元素 ≈ 0（无相关性）

---

## 🔍 根本原因

### 代码问题
在 `_fit_tfidf_svd()` 和 `_fit_on_train_transform_all()` 函数中：

```python
# 第157-164行
vectorizer = TfidfVectorizer(
    ...,
    norm=None,  # 不在TF-IDF阶段归一化（正确）
    ...
)

# 第167-168行 - 问题所在！
if pre_svd_l2:  # 默认值 = True
    tfidf = l2_normalize(tfidf, norm="l2", axis=1, copy=False)
```

### 为什么这导致白化失败？

1. **L2归一化改变协方差结构**
   - 原始TF-IDF向量的协方差反映真实的方差分布
   - L2归一化后，所有向量都在单位球面上
   - 协方差矩阵的特征值分布被扭曲

2. **SVD在"错误"的空间中降维**
   - SVD基于L2归一化后的向量
   - 降维结果保留的是"扭曲"后的主成分
   - 不是真实数据的主方向

3. **白化基于"不自然"的协方差**
   - 白化矩阵计算：`W = U @ diag(1/sqrt(S))`，其中S是协方差的特征值
   - 但协方差已经不反映真实数据方差
   - 结果：对角线≠1.0，白化不完全

### 数学解释

设原始向量为 $\mathbf{x}_i$，L2归一化后为 $\mathbf{x}'_i = \frac{\mathbf{x}_i}{||\mathbf{x}_i||}$

- **原始协方差**: $\text{Cov}(\mathbf{X}) = \frac{1}{n}\sum_i \mathbf{x}_i\mathbf{x}_i^T$
- **归一化后**: $\text{Cov}(\mathbf{X}') = \frac{1}{n}\sum_i \frac{\mathbf{x}_i\mathbf{x}_i^T}{||\mathbf{x}_i||^2}$（权重不均！）

白化应基于原始协方差结构，而非归一化后的！

---

## ✅ 修复方案

### 代码修改

**修改1**: `_fit_tfidf_svd()` 函数（第148行）
```python
# 修改前
pre_svd_l2: bool = True,

# 修改后
pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
```

**修改2**: `_fit_on_train_transform_all()` 函数（第286行）
```python
# 修改前
pre_svd_l2: bool = True,

# 修改后
pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
```

**修改3**: `build_item_text_emb()` 函数（第350行）
```python
# 修改前
pre_svd_l2: bool = True,

# 修改后
pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
```

### 文档更新

更新了函数文档字符串，说明：
```python
IMPORTANT: 
- pre_svd_l2 is now False by default to preserve natural covariance structure
- This allows subsequent whitening to work correctly (Cov → I)
- L2 normalization (if needed) should be done AFTER whitening, not before SVD
```

---

## 📊 预期改进

### 修复前（pre_svd_l2=True）
```
对角线均值: 0.8418
对角线标准差: 0.1682
非对角线平均值: 0.0675
非对角线最大值: 0.6651
结论: ❌ 白化不完全
```

### 修复后（pre_svd_l2=False）
```
对角线均值: 0.95-1.05  ✅
对角线标准差: <0.1     ✅
非对角线平均值: <0.05   ✅
非对角线最大值: <0.3    ✅
结论: ✅ 白化成功
```

---

## 🚀 使用指南

### 重新生成embeddings（推荐）

```bash
# 使用新的默认设置（pre_svd_l2=False）
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256

# 验证白化效果
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
```

### 向后兼容

如果需要旧的行为（例如对比实验）：
```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.old.npy \
  --svd_dim 256 \
  --pre_svd_l2  # 显式启用（注意：无下划线，argparse会转换）
```

注意：命令行参数是 `--pre_svd_l2`（不是 `--no_pre_svd_l2`），因为默认已经是False了。

---

## 🔬 验证步骤

### 1. 重新生成embeddings
```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256
```

### 2. 验证白化效果
```bash
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
```

### 3. 检查输出
期望看到：
```
✅ Embedding已中心化 (mean_abs<0.2)
📊 协方差矩阵分析:
   - 对角线均值: 0.95-1.05 (期望≈1.0)  ✅
   - 对角线标准差: <0.1 (期望≈0)       ✅
   - 非对角线平均值: <0.05 (期望≈0)     ✅
✅ 检测到良好的白化效果
✅ 当前文件验证通过
```

---

## 📝 技术细节

### 为什么TF-IDF不需要pre-SVD归一化？

1. **TF-IDF本身已稀疏归一化**
   - `TfidfVectorizer(norm=None)` 虽然禁用了L2归一化
   - 但TF-IDF权重已经是归一化的（IDF = log(N/df)）
   - 向量模长差异反映文档长度/重要性差异

2. **SVD的数值稳定性**
   - TruncatedSVD内部使用randomized算法，数值稳定
   - 稀疏矩阵的SVD不需要预归一化
   - sklearn的实现已经处理了数值问题

3. **保留方差信息**
   - 不同向量的模长差异 = 方差信息
   - SVD应该保留这些差异（主成分分析）
   - 白化会基于真实方差进行标准化

### 白化的正确流程

```
原始向量 → TF-IDF → SVD降维 → 中心化 → 白化 → [可选: L2归一化]
   ↑                   ↑              ↑        ↑
   保留              保留           去均值    去相关+
   语义              方差                     标准化方差
```

**关键**：不要在SVD前做L2归一化，否则方差信息丢失！

---

## ⚠️ 影响范围

### 受影响的功能
- ✅ TF-IDF baseline embeddings 白化效果改进
- ✅ 所有使用 `build_item_text_emb_base.py` 生成的特征

### 不受影响的功能
- ✅ Qwen3 embeddings（使用不同的脚本）
- ✅ 已生成的旧embeddings（除非重新生成）
- ✅ 模型训练逻辑（embedding只是输入）

### 建议行动
1. **重新生成所有TF-IDF embeddings**（所有数据集）
2. **重新验证白化效果**
3. **可选：重新运行实验对比性能**（预期：白化改进后性能可能提升）

---

## 📚 相关文档

- `tools/verify_whiten.py` - 白化效果验证脚本
- `tools/README_WHITEN.md` - 白化技术文档
- `fix_whiten_analysis.md` - 详细问题分析

---

## ✅ Checklist

- [x] 修改 `_fit_tfidf_svd()` 默认参数
- [x] 修改 `_fit_on_train_transform_all()` 默认参数
- [x] 修改 `build_item_text_emb()` 默认参数
- [x] 更新函数文档字符串
- [x] 创建changelog文档
- [ ] 重新生成所有数据集的TF-IDF embeddings
- [ ] 验证所有生成的embeddings
- [ ] 对比修复前后的模型性能

---

**修复者**: Assistant  
**审核者**: lvchao0428  
**状态**: ✅ 代码修复完成，等待重新生成和验证

