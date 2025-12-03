# 🔧 关键修复：SVD + Whitening 顺序问题

## 🚨 问题描述

用户在验证TF-IDF特征时发现白化失败：

```bash
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

📊 协方差矩阵分析 (sample_size=500):
   - 对角线均值: 0.0039 (期望≈1.0)  ← ❌ 问题！
   - 对角线标准差: 0.0014 (期望≈0)
   - 非对角线最大值: 0.0074 (期望≈0)
   - 非对角线平均值: 0.0004 (期望≈0)
❌ 未检测到白化效果
```

**期望值：** 白化后的协方差矩阵对角线应接近 1.0，非对角线接近 0  
**实际值：** 对角线仅 0.0039，说明白化完全失效

---

## 🔍 根本原因

### 错误的处理顺序

```python
# ❌ 错误流程（修复前）
1. TF-IDF vectorization
2. TruncatedSVD 降维
3. L2 normalize  ← 问题所在！
4. Center + Whiten + L2 normalize  ← 白化失败
```

**问题分析：**

1. **Step 3** 的 L2 归一化将所有向量强制到**单位球面**上
2. 每个向量的范数都变成 1.0，数据失去了**自然的方差结构**
3. **Step 4** 的白化期望数据有自然的方差分布
4. 白化计算协方差矩阵时，数据已经被归一化，导致：
   - 协方差矩阵的特征值异常小（0.0039 vs 1.0）
   - 白化变换无法正确去相关化

### 数学原理

**白化（Whitening）的前提假设：**
- 数据 X 有自然的协方差结构：Cov(X) = Σ
- 白化目标：找到变换 W，使得 Cov(WX) = I（单位矩阵）

**L2归一化的影响：**
- L2归一化后：||x|| = 1（所有向量在单位球面上）
- 此时数据的方差主要由角度差异决定，而非幅度
- 协方差矩阵的scale被破坏：diag(Cov) << 1

---

## ✅ 解决方案

### 正确的处理顺序

```python
# ✅ 正确流程（修复后）
1. TF-IDF vectorization
2. (Optional) Pre-SVD L2 normalize  ← 仅用于数值稳定性
3. TruncatedSVD 降维
4. 保留原始 scale（不做 L2 归一化）
5. Center + Whiten + L2 normalize  ← 白化成功
```

**关键改进：**
- 移除 SVD 后的 L2 归一化
- 让数据保持自然的方差结构
- 白化步骤内部会正确计算协方差并归一化

---

## 📝 代码修改

### 1. `build_item_text_emb_base.py`

#### 修改函数：`_fit_tfidf_svd` 和 `_fit_on_train_transform_all`

**修改前：**
```python
def _fit_on_train_transform_all(...):
    # ... TF-IDF and SVD ...
    reduced = svd.transform(tfidf_all)
    
    # ❌ 错误：SVD后立即L2归一化
    reduced = l2_normalize(reduced, norm="l2", axis=1, copy=False)
    reduced[0, :] = 0.0
    return reduced.astype(np.float32)
```

**修改后：**
```python
def _fit_on_train_transform_all(...):
    """...
    
    NOTE: This function does NOT apply L2 normalization after SVD.
    The caller should apply center+whiten+L2 normalization afterwards.
    """
    # ... TF-IDF and SVD ...
    reduced = svd.transform(tfidf_all)
    
    # ✅ 正确：保留原始scale，不做L2归一化
    # Do NOT L2 normalize here - let the whitening step handle normalization
    # This preserves the natural variance structure needed for whitening
    reduced[0, :] = 0.0  # Ensure PAD row is zeros
    return reduced.astype(np.float32)
```

---

### 2. `build_item_text_emb_qwen3_hf.py`

#### 修改函数：`_apply_truncated_svd`

**添加参数控制归一化：**

```python
def _apply_truncated_svd(
    mat: np.ndarray, 
    target_dim: int, 
    train_ids, 
    random_state: int, 
    label: str,
    normalize: bool = True  # ← 新增参数
):
    """Apply TruncatedSVD dimensionality reduction.
    
    Args:
        normalize: Whether to L2 normalize after SVD (default: True)
                  Set to False if whitening will be applied afterwards.
    """
    # ... SVD logic ...
    
    # Optionally normalize after SVD
    if normalize:
        reduced = l2_normalize(reduced, norm="l2", axis=1)
    
    projected[1:, :] = reduced
    return projected
```

#### 修改调用点：

**Per-view SVD（用于保存分视图）：**
```python
# 不归一化，保留方差结构用于后续白化
view_mat = _apply_truncated_svd(
    view_mat,
    target_dim=args.view_project_dim,
    normalize=False,  # ← 保留方差结构
)
```

**Final SVD（白化之后的可选降维）：**
```python
# 白化之后的SVD可以归一化
mat = _apply_truncated_svd(
    mat,
    target_dim=args.project_dim,
    normalize=True,  # ← 重新归一化OK
)
```

---

## 📊 修复前后对比

### 修复前（失败）

```bash
✅ 加载embedding: item_text_emb.base.npy
   - 形状: (259205, 256)
   - 数据类型: float16
✅ PAD embedding 为零向量 (norm=0.00e+00)
⚠️  Embedding未充分L2归一化 (mean=1.0000, std=inf)  ← L2归一化两次导致数值问题
✅ Embedding已中心化 (mean_abs=0.0076)

📊 协方差矩阵分析:
   - 对角线均值: 0.0039 (期望≈1.0)  ← ❌ 白化失败
   - 非对角线最大值: 0.0074
❌ 未检测到白化效果
```

### 修复后（预期）

```bash
✅ 加载embedding: item_text_emb.base.npy
   - 形状: (259205, 256)
   - 数据类型: float16
✅ PAD embedding 为零向量 (norm=0.00e+00)
✅ Embedding已L2归一化 (mean=0.9998, std=0.0012)  ← 正常
✅ Embedding已中心化 (mean_abs=0.0234)

📊 协方差矩阵分析:
   - 对角线均值: 1.0023 (期望≈1.0)  ← ✅ 白化成功
   - 对角线标准差: 0.0456 (期望≈0)
   - 非对角线最大值: 0.1234 (期望≈0)
   - 非对角线平均值: 0.0089 (期望≈0)
✅ 白化效果良好
```

---

## 🎯 影响范围

### 需要重新生成的特征

所有使用旧版本脚本生成的特征都需要重新生成：

```bash
# 1. TF-IDF特征
bash tools/gen_tfidf_only.sh

# 2. Qwen3单视图
python tools/build_item_text_emb_qwen3_hf.py \
  --prompt_preset base \
  --output_mode mean \
  ...

# 3. Qwen3多视图
bash tools/gen_qwen3_multiview_only.sh

# 4. 完整版（推荐）
bash tools/gen_text_emb_beauty_full.sh
```

### 验证修复效果

```bash
# 重新生成后验证
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# 或批量验证
bash tools/verify_all_embeddings.sh
```

---

## 🔬 技术细节

### Pre-SVD L2 归一化 vs Post-SVD L2 归一化

| 操作 | 目的 | 是否保留 |
|------|------|---------|
| **Pre-SVD L2** | 数值稳定性，确保 SVD 输入范围一致 | ✅ 保留 |
| **Post-SVD L2** | 强制输出到单位球面 | ❌ 移除（与whitening冲突） |

### 白化变换的数学

白化变换计算：

```python
# 1. Center
X_centered = X - mean(X_train)

# 2. 计算协方差矩阵（仅训练集）
Cov = (X_train_centered.T @ X_train_centered) / n_train

# 3. SVD分解协方差矩阵
U, S, _ = svd(Cov)

# 4. 白化矩阵
W = U @ diag(1 / sqrt(S + eps))

# 5. 应用白化
X_whitened = X_centered @ W

# 6. L2 归一化
X_final = X_whitened / ||X_whitened||
```

**关键点：**
- 如果 X 已经被 L2 归一化，`Cov` 的特征值会异常小
- 导致 `1/sqrt(S)` 异常大，白化变换不稳定

---

## 📚 相关文档

- [TEXT_EMB_GENERATION_GUIDE.md](./TEXT_EMB_GENERATION_GUIDE.md) - 特征生成指南
- [VERIFICATION_GUIDE.md](./VERIFICATION_GUIDE.md) - 验证指南
- [CHANGELOG_VERIFY_WHITEN.md](./CHANGELOG_VERIFY_WHITEN.md) - 验证工具改进记录

---

## ✅ 检查清单

重新生成特征后，确认：

- [ ] 重新运行特征生成脚本
- [ ] 验证白化效果：`bash tools/verify_all_embeddings.sh`
- [ ] 确认协方差矩阵对角线接近 1.0
- [ ] 确认非对角线元素接近 0
- [ ] 确认 L2 归一化正常（mean ≈ 1.0, std < 0.01）
- [ ] 重新运行实验：`bash two_phase_run_tfidf.sh`
- [ ] 对比新旧特征的实验结果

---

## 🐛 如果验证仍然失败

### 可能的原因

1. **数据集太小**
   - 训练集样本不足，协方差估计不稳定
   - 解决：检查数据集大小

2. **数据方差太小**
   - 所有文本相似度过高
   - 解决：检查文本多样性

3. **float16 精度问题**
   - 某些情况下 float16 精度不够
   - 解决：使用 `--dtype float32`

### 调试步骤

```bash
# 1. 使用 float32 重新生成
python tools/build_item_text_emb_base.py \
  --dtype float32 \
  ...

# 2. 检查统计量文件
python -c "
import numpy as np
stats = np.load('dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz')
print('Mean shape:', stats['mean'].shape)
print('Mean norm:', np.linalg.norm(stats['mean']))
if 'whiten_matrix' in stats:
    print('Whiten matrix shape:', stats['whiten_matrix'].shape)
    print('Whiten matrix condition:', np.linalg.cond(stats['whiten_matrix']))
"

# 3. 检查原始embedding（whiten之前）
# 需要临时修改代码，在whiten之前保存一份
```

---

**修复日期**: 2025-12-03  
**版本**: v2.1  
**影响**: 所有特征需重新生成  
**优先级**: 🔴 HIGH - 直接影响模型性能

