# Qwen3 白化问题修复总结

## 问题现象

验证 `dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy` 时白化效果检测失败：

```
✅ Embedding已L2归一化 (mean=1.0000, std=0.0000)
✅ Embedding已中心化 (mean_abs=0.0011)
❌ 对角线均值: 0.0039 (期望≈1.0)  ← 核心问题
   - 应该是 1.0，实际只有 0.0039 (256倍差距)
```

## 根本原因

**白化后又做了 L2 归一化，破坏了白化的本质目的**

### 数学原理

1. **白化目标**: Cov(X) = I → 每个维度方差 = 1.0
2. **L2 归一化**: ||x|| = 1 → 每个维度方差 ≈ 1/D
3. **实际结果**: 256 维向量，1/256 = 0.00390625 ≈ 0.0039 ✗

### 代码对比

**TF-IDF 基线** (`build_item_text_emb_base.py`): ✅ 正确
```python
# NOTE: Do NOT L2 normalize after whitening!
# Whitening already decorrelates features and sets Cov(X) = I
```

**Qwen3 脚本** (`build_item_text_emb_qwen3_hf.py`): ❌ 错误（已修复）
```python
# NOTE: We MUST L2 normalize even after whitening for RecBole compatibility.
norms = np.linalg.norm(emb_whitened[1:], axis=1, keepdims=True)
emb_whitened[1:] = emb_whitened[1:] / np.clip(norms, 1e-8, None)
```

## 修复内容

### ✅ 修复点 1: 移除白化后的 L2 归一化

**文件**: `tools/build_item_text_emb_qwen3_hf.py`  
**位置**: 第 324-335 行（`_center_whiten_and_normalize` 函数）

删除了白化后的归一化代码，与 TF-IDF 基线保持一致。

### ✅ 修复点 2: SVD 投影不归一化

**文件**: `tools/build_item_text_emb_qwen3_hf.py`  
**位置**: 第 680 行

```python
# 修改前
normalize=True,  # Re-normalize after whitening+projection

# 修改后  
normalize=False,  # Preserve variance for subsequent whitening
```

### ✅ 修复验证

运行 `python3 tools/verify_whiten_fix.py` 确认：
```
✅ 修复点1已应用: 移除了白化后的 L2 归一化
✅ 修复点2已应用: SVD 投影不再归一化
✅ 所有修复已正确应用！
```

## 影响范围

### 需要重新生成的文件

**Amazon_Beauty**:
- ❌ `dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy` (单视图 256维)
- ❌ `dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz`
- ❌ `dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy` (拼接 256维)
- ❌ `dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz`
- ❌ `dataset/Amazon_Beauty/qwen3_4views/view_0.npy` (64维，对角线≈0.0156)
- ❌ `dataset/Amazon_Beauty/qwen3_4views/view_1.npy` (64维，对角线≈0.0156)
- ❌ `dataset/Amazon_Beauty/qwen3_4views/view_2.npy` (64维，对角线≈0.0156)
- ❌ `dataset/Amazon_Beauty/qwen3_4views/view_3.npy` (64维，对角线≈0.0156)
- ❌ `dataset/Amazon_Beauty/qwen3_4views/view_*_whiten_stats.npz` (4个统计文件)

**注**: Multi-view 有两层白化（每个视图 + 拼接后），都受此问题影响

**其他数据集**: 如果已生成 Qwen3 特征，也需要重新生成

### 不受影响的文件

- ✅ `dataset/Amazon_Beauty/item_text_emb.base.npy` (TF-IDF，无需重新生成)
- ✅ `dataset/Amazon_Beauty/item_index_mapping.csv`

## 下一步操作

### 1. 重新生成 Qwen3 特征

**方法 A: 重新生成所有 Qwen3 特征**（推荐，单视图+多视图）

```bash
cd /home/charlie/project/RecBole
bash tools/regen_all_qwen3_beauty.sh
```

这会重新生成：
- 单视图特征 (`item_text_emb.qwen3.base.npy`, 256维)
- Multi-view 特征 (`qwen3_4views/view_*.npy`, 4×64维)
- 拼接特征 (`item_text_emb.qwen3.multiview.npy`, 256维)
- 所有统计文件
- 并自动验证白化效果

**方法 B: 仅重新生成单视图**（快速，仅单视图）

```bash
cd /home/charlie/project/RecBole
bash tools/regen_qwen3_beauty.sh
```

**方法 C: 仅重新生成 Multi-view**（仅多视图）

```bash
cd /home/charlie/project/RecBole
bash tools/regen_multiview_beauty.sh
```

**方法 D: 重新生成全部特征**（TF-IDF + Qwen3，最完整但最耗时）

```bash
cd /home/charlie/project/RecBole
bash tools/gen_text_emb_beauty_full_fast.sh
```

### 2. 验证白化效果

**验证单视图特征 (256维)**:
```bash
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --dataset Amazon_Beauty
```

**验证 Multi-view 各视图 (64维)**:
```bash
# 验证视图0
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/qwen3_4views/view_0.npy \
  --dataset Amazon_Beauty

# 同理验证 view_1.npy, view_2.npy, view_3.npy
```

**期望输出**:
```
✅ PAD embedding 为零向量
ℹ️  Embedding未L2归一化 (mean=16.xxxx, std=x.xxxx)  ← 正常！
   提示: Whitened embeddings通常不做L2归一化（这是正常的）

✅ Embedding已中心化 (mean_abs<0.2)

📊 协方差矩阵分析:
   - 对角线均值: 0.95~1.05 (期望≈1.0)  ← 修复成功！
   - 对角线标准差: <0.15 (期望≈0)
   - 非对角线最大值: <0.05 (期望≈0)
   - 非对角线平均值: <0.05 (期望≈0)

✅ 白化效果优秀
✅ 当前文件验证通过
```

**注意向量范数**: 
- 单视图 (256维): `mean ≈ 16` (sqrt(256) = 16)
- Multi-view 视图 (64维): `mean ≈ 8` (sqrt(64) = 8)
- **未L2归一化是正常的**，这是修复后的预期行为

### 3. 其他数据集（如需要）

```bash
# Toys
bash tools/gen_text_emb_toys_full_fast.sh

# VideoGames
bash tools/gen_text_emb_videogames_full_fast.sh

# Yelp
bash tools/gen_text_emb_yelp_full_fast.sh
```

## 技术细节

### 修复前的处理流程（错误）

```
Qwen3 Embeddings (4096维)
  ↓ L2归一化
归一化向量 (4096维, norm=1)
  ↓ output_mode=mean
平均向量 (4096维, norm=1)
  ↓ SVD投影 + L2归一化  ← 问题1
降维向量 (256维, norm=1)
  ↓ 白化 + L2归一化  ← 问题2
错误结果 (256维, Cov对角线≈0.0039) ✗
```

### 修复后的处理流程（正确）

```
Qwen3 Embeddings (4096维)
  ↓ L2归一化
归一化向量 (4096维, norm=1)
  ↓ output_mode=mean
平均向量 (4096维, norm=1)
  ↓ SVD投影（不归一化）← 修复
降维向量 (256维, norm≈16)
  ↓ 白化（不归一化）← 修复
正确结果 (256维, Cov≈I, 对角线≈1.0) ✓
```

## 潜在影响

### 模型性能

- **可能提升**: 正确的白化可能改善特征质量，提升模型性能
- **需要重新训练**: 使用旧特征训练的模型需要用新特征重新训练
- **对比实验**: 建议对比修复前后的模型性能

### 向量范数

- **修复前**: 所有向量 L2 范数 = 1
- **修复后**: 向量范数 ≈ sqrt(D) ≈ 16（对于 256 维）
- **点积范围**: 从 [-1, 1] 变为约 [-256, 256]
- **温度参数**: 模型的温度参数可能需要调整以适应新的点积范围

### 兼容性

- ✅ RecBole 完全兼容（无需修改模型代码）
- ✅ 与 TF-IDF 基线白化逻辑完全一致
- ⚠️  旧模型 checkpoint 不兼容新特征（需要重新训练）

## 相关文档

- `WHITEN_FIX_QWEN3.md` - 详细的修复说明和代码对比
- `tools/verify_whiten_fix.py` - 代码修复验证工具
- `tools/regen_qwen3_beauty.sh` - 快速重新生成脚本
- `tools/verify_whiten.py` - 白化效果验证工具

## 修复日期

2025-12-05

## 修复状态

✅ **代码修复完成**  
⏳ **等待重新生成特征**  
⏳ **等待验证结果**

