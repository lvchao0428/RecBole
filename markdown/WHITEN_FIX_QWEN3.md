# Qwen3 特征白化问题修复说明

## 问题诊断

验证 Qwen3 特征时发现白化失败：
```
✅ Embedding已L2归一化 (mean=1.0000, std=0.0000)
✅ Embedding已中心化 (mean_abs=0.0011)
❌ 对角线均值: 0.0039 (期望≈1.0)  ← 问题！
```

### 根本原因

**白化后又做了 L2 归一化，破坏了白化效果**

- **白化目的**: 使协方差矩阵变为单位矩阵 Cov(X) = I，每个维度方差 ≈ 1.0
- **L2 归一化影响**: 将向量归一化为单位长度，导致每个维度方差 ≈ 1/D
- **实际结果**: 对于 256 维，方差 = 1/256 ≈ 0.0039 ✗

## 代码修复

### 修复点 1: 移除白化后的 L2 归一化

**文件**: `tools/build_item_text_emb_qwen3_hf.py`

**修改位置**: 第 324-335 行（`_center_whiten_and_normalize` 函数）

**修改前**:
```python
# Step 3: Apply whitening to all embeddings
emb_whitened = emb_centered @ whiten_matrix
emb_whitened[0, :] = 0.0

# NOTE: We MUST L2 normalize even after whitening for RecBole compatibility.
# While strictly speaking this distorts the identity covariance, 
# it is necessary to keep dot products in a reasonable range for InfoNCE/Temperature.
# Theoretical norm of whitened vector is sqrt(D), which is too large (e.g. sqrt(4096)=64).
norms = np.linalg.norm(emb_whitened[1:], axis=1, keepdims=True)
emb_whitened[1:] = emb_whitened[1:] / np.clip(norms, 1e-8, None)

emb_processed = emb_whitened.astype(np.float32)  # Cast back to float32
```

**修改后**:
```python
# Step 3: Apply whitening to all embeddings
emb_whitened = emb_centered @ whiten_matrix
emb_whitened[0, :] = 0.0

# NOTE: Do NOT L2 normalize after whitening!
# Whitening already decorrelates features and sets Cov(X) = I
# L2 normalization would destroy this property (variance becomes 1/D instead of 1)
# If the model needs L2-normalized embeddings, do it at model load time

emb_processed = emb_whitened.astype(np.float32)  # Cast back to float32
```

### 修复点 2: SVD 投影时不归一化

**文件**: `tools/build_item_text_emb_qwen3_hf.py`

**修改位置**: 第 669-681 行（主函数的 SVD 投影部分）

**修改前**:
```python
# Final SVD after whitening - normalize is OK here
mat = _apply_truncated_svd(
    mat,
    target_dim=args.project_dim,
    train_ids=train_ids_cache,
    random_state=args.svd_random_state,
    label="final",
    normalize=True,  # Re-normalize after whitening+projection
)
```

**修改后**:
```python
# SVD projection BEFORE whitening - do NOT normalize to preserve variance
mat = _apply_truncated_svd(
    mat,
    target_dim=args.project_dim,
    train_ids=train_ids_cache,
    random_state=args.svd_random_state,
    label="final",
    normalize=False,  # Preserve variance for subsequent whitening
)
```

## 与 TF-IDF 基线的一致性

修复后，Qwen3 的白化逻辑与 TF-IDF 基线（`build_item_text_emb_base.py`）完全一致：

**TF-IDF 基线**（第 250-253 行）:
```python
# NOTE: Do NOT L2 normalize after whitening!
# Whitening already decorrelates features and sets Cov(X) = I
# L2 normalization would destroy this property
```

## 修复后的处理流程

```
原始 Qwen3 Embeddings (4096 维)
  ↓ L2 归一化（encode_batch 内部）
原始归一化向量
  ↓ output_mode=mean: 平均后再归一化
聚合向量 (4096 维)
  ↓ SVD 投影（不归一化）← 修复点 2
降维向量 (256 维)
  ↓ 白化: Center + Whiten（不归一化）← 修复点 1
最终向量 (256 维, Cov = I, 方差 ≈ 1.0) ✓
```

## 重新生成特征

### 方法 1: 使用提供的快速脚本

```bash
cd /home/charlie/project/RecBole
bash tools/regen_qwen3_beauty.sh
```

### 方法 2: 手动运行生成命令

```bash
cd /home/charlie/project/RecBole

# 备份旧文件
mv dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
   dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy.backup

mv dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz \
   dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz.backup

# 重新生成
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

# 验证白化效果
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --dataset Amazon_Beauty
```

### 方法 3: 运行完整的生成脚本

```bash
cd /home/charlie/project/RecBole
bash tools/gen_text_emb_beauty_full_fast.sh
```

注意：这会重新生成所有特征（TF-IDF + Qwen3 单视图 + Qwen3 多视图），耗时较长。

## 验证期望结果

修复后运行验证应该看到：

```
======================================================================
验证 Center + Whiten 预处理
======================================================================

✅ 加载embedding: item_text_emb.qwen3.base.npy
   - 形状: (259205, 256)
   - 数据类型: float16

✅ 找到统计量文件: item_text_emb.qwen3.base_whiten_stats.npz
   - mean shape: (1, 4096)
   - whiten_matrix shape: (4096, 4096)

✅ PAD embedding 为零向量

ℹ️  Embedding未L2归一化 (mean=16.0000, std=0.xxxx)  ← 修复后不再归一化
   提示: Whitened embeddings通常不做L2归一化（这是正常的）

✅ Embedding已中心化 (mean_abs<0.2)

📊 协方差矩阵分析 (sample_size=2000):
   - 对角线均值: 0.95~1.05 (期望≈1.0)  ← 修复成功！
   - 对角线标准差: <0.15 (期望≈0)
   - 非对角线最大值: <0.05 (期望≈0)
   - 非对角线平均值: <0.05 (期望≈0)

✅ 白化效果优秀  ← 期望结果
✅ 当前文件验证通过
```

## 其他数据集

如果其他数据集（Toys, VideoGames, Yelp）也需要重新生成，使用相应的脚本：

```bash
# Toys
bash tools/gen_text_emb_toys_full_fast.sh

# VideoGames
bash tools/gen_text_emb_videogames_full_fast.sh

# Yelp
bash tools/gen_text_emb_yelp_full_fast.sh
```

## 注意事项

1. **必须重新生成特征**：代码修复后，之前生成的 Qwen3 特征都是错误的，必须重新生成
2. **TF-IDF 基线无需重新生成**：TF-IDF 的白化实现一直是正确的
3. **多视图特征也需要重新生成**：`item_text_emb.qwen3.multiview.npy` 也受此问题影响
4. **模型训练结果可能改善**：修复后的白化可能提升模型性能

## 相关文件

- `tools/build_item_text_emb_qwen3_hf.py` - Qwen3 特征生成脚本（已修复）
- `tools/verify_whiten.py` - 白化验证工具
- `tools/gen_text_emb_beauty_full_fast.sh` - Beauty 完整生成脚本
- `tools/regen_qwen3_beauty.sh` - Beauty Qwen3 快速重新生成脚本（新建）

