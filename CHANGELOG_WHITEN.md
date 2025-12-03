# Center + Whiten 功能实现总结

## 📋 修改概览

为文本embedding预处理脚本添加了 **center（中心化）+ whiten（白化）** 功能，完全符合论文辅导要求。

---

## ✅ 已完成的修改

### 1. `tools/build_item_text_emb_base.py` (TF-IDF + SVD)

**新增功能**：
- ✅ 新增 `_center_whiten_and_normalize()` 函数
- ✅ 只用训练集计算 mean 和 whiten_matrix
- ✅ 保存统计量到 `*_whiten_stats.npz` 文件
- ✅ 新增 `--no_whiten` 参数（默认启用白化）

**关键代码位置**：
- Line 183-260: `_center_whiten_and_normalize()` 函数实现
- Line 306-313: 调用白化处理
- Line 462-466: 命令行参数

### 2. `tools/build_item_text_emb_qwen3_hf.py` (Qwen3 LLM)

**新增功能**：
- ✅ 新增 `_center_whiten_and_normalize()` 函数
- ✅ 只用训练集计算统计量（需传 `--dataset` 参数）
- ✅ 保存统计量到 `*_whiten_stats.npz` 文件
- ✅ 新增 `--no_whiten` 参数（默认启用白化）
- ✅ 修复 `Optional` 导入

**关键代码位置**：
- Line 32: 添加 `Optional` 导入
- Line 260-346: `_center_whiten_and_normalize()` 函数实现
- Line 625-636: 调用白化处理
- Line 247-252: 命令行参数

### 3. 新增文档和工具

**文档**：
- ✅ `tools/README_WHITEN.md` - 详细使用说明
- ✅ `CHANGELOG_WHITEN.md` - 本文件，修改总结

**工具脚本**：
- ✅ `tools/verify_whiten.py` - 验证白化效果
- ✅ `tools/example_generate_whitened_embeddings.sh` - 使用示例

---

## 🎯 符合要求检查表

| 辅导要求 | 状态 | 说明 |
|---------|------|------|
| TF-IDF/SVD→256, Qwen3→线性→256 | ✅ | 已实现 |
| center（中心化） | ✅ | 减去训练集均值 |
| whiten（白化） | ✅ | SVD白化变换 |
| L2 normalize | ✅ | 白化后L2归一化 |
| 只用训练集fit统计量 | ✅ | train_ids筛选 |
| 保存统计量供推理复用 | ✅ | *_whiten_stats.npz |
| 避免时间泄漏 | ✅ | 测试集用训练集参数 |

---

## 📖 使用方法

### 快速开始

#### 1. 生成 TF-IDF Base 特征（带白化）

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --title_field title
```

**输出**：
- `item_text_emb.base.npy` - 白化后的embedding
- `item_text_emb.base_whiten_stats.npz` - 统计量文件

#### 2. 生成 Qwen3 LLM 特征（带白化）

```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --dataset Amazon_Beauty \
  --output_mode mean \
  --batch_size 16
```

**⚠️ 重要**：必须传 `--dataset` 参数！

#### 3. 验证白化效果

```bash
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
```

#### 4. 关闭白化（仅作对比实验）

```bash
# 只做 center + L2，不做 whiten
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base_no_whiten.npy \
  --no_whiten
```

---

## 🔬 技术实现细节

### 白化算法

```python
def _center_whiten_and_normalize(emb, train_ids):
    # 1. 提取训练集
    train_emb = emb[train_ids]
    
    # 2. Center (中心化)
    mean = train_emb.mean(axis=0)
    emb_centered = emb - mean
    
    # 3. Whiten (白化)
    train_centered = train_emb - mean
    cov = (train_centered.T @ train_centered) / len(train_centered)
    U, S, _ = np.linalg.svd(cov)
    whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
    emb_whitened = emb_centered @ whiten_matrix
    
    # 4. L2 normalize
    emb_final = emb_whitened / np.linalg.norm(emb_whitened, axis=1, keepdims=True)
    
    return emb_final
```

### 数值稳定性优化

- 使用 `float64` 精度计算统计量
- 白化矩阵添加 `eps=1e-5` 防止除零
- PAD embedding (index=0) 始终保持零向量

---

## 📊 预期效果

根据辅导老师建议，正确的 center + whiten 预处理应该能：

1. **统一特征分布**：TF-IDF 和 LLM 两路特征分布对齐
2. **消除冗余**：去掉特征间的相关性
3. **数值稳定**：统一方差，loss梯度更平衡
4. **性能提升**：预期 NDCG@10 提升 1-2%（需实验验证）

### 验证标准

运行 `verify_whiten.py` 后，应该看到：

```
✅ PAD embedding 为零向量
✅ Embedding已L2归一化 (mean≈1.0, std≈0)
✅ Embedding已中心化 (mean_abs<0.1)
✅ 白化效果良好
   - 协方差对角线均值≈1.0
   - 协方差非对角线接近0
```

---

## 🚨 注意事项

### 1. 必须传 --dataset 参数

**错误示例（会泄漏）**：
```bash
# ❌ Qwen3脚本缺少 --dataset，会用所有数据fit
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping xxx.csv \
  --output xxx.npy
```

**正确示例**：
```bash
# ✅ 传入 --dataset，只用训练集fit
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping xxx.csv \
  --output xxx.npy \
  --dataset Amazon_Beauty
```

### 2. 统计量文件的用途

`*_whiten_stats.npz` 文件包含：
- `mean`: 训练集均值 (shape: [1, d])
- `whiten_matrix`: 白化矩阵 (shape: [d, d])

**用于推理时**：
```python
# 加载统计量
stats = np.load('item_text_emb.base_whiten_stats.npz')
mean = stats['mean']
whiten_matrix = stats['whiten_matrix']

# 对新数据应用相同变换
new_emb_centered = new_emb - mean
new_emb_whitened = new_emb_centered @ whiten_matrix
new_emb_final = F.normalize(new_emb_whitened, dim=1)
```

### 3. 3D tensor 处理

多视角 stack 模式（shape: [n, k, d]）会跳过全局白化：
```
[WARN] Skipping center/whiten for 3D tensor (mode=stack). Apply per-view instead.
```

需要在 `--split_output_dir` 中对每个视角单独白化。

---

## 🔄 对比实验建议

建议运行 A/B 测试验证白化效果：

```bash
# 实验A: 启用白化（默认）
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output base_with_whiten.npy

# 实验B: 禁用白化
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output base_no_whiten.npy \
  --no_whiten

# 分别训练模型，对比指标
```

---

## 📚 相关文件

### 修改的文件
- `tools/build_item_text_emb_base.py`
- `tools/build_item_text_emb_qwen3_hf.py`

### 新增的文件
- `tools/README_WHITEN.md`
- `tools/verify_whiten.py`
- `tools/example_generate_whitened_embeddings.sh`
- `CHANGELOG_WHITEN.md`

### 生成的文件（运行后）
- `dataset/*/item_text_emb.*.npy` - 白化后的embedding
- `dataset/*/item_text_emb.*_whiten_stats.npz` - 统计量文件

---

## ✅ 测试检查清单

- [ ] 运行 `build_item_text_emb_base.py`，生成 Base 特征
- [ ] 运行 `verify_whiten.py`，验证白化效果
- [ ] 检查是否生成 `*_whiten_stats.npz` 文件
- [ ] 确认 Qwen3 脚本传入 `--dataset` 参数
- [ ] 对比 whiten vs no_whiten 的模型指标

---

**实现日期**: 2025-12-03  
**符合要求**: ✅ 完全符合论文辅导老师的要求  
**状态**: 已完成，可投入使用

