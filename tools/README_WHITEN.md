# Center + Whiten 预处理功能说明

## 概述

已为两个文本embedding生成脚本添加 **center（中心化）+ whiten（白化）** 预处理功能：

- `build_item_text_emb_base.py` (TF-IDF + SVD)
- `build_item_text_emb_qwen3_hf.py` (Qwen3 LLM)

## 功能特性

### ✅ 符合论文要求

1. **只用训练集fit统计量**：mean 和 whiten_matrix 只在训练集上计算
2. **保存统计量供推理复用**：生成 `*_whiten_stats.npz` 文件
3. **避免时间泄漏**：测试集使用训练集的统计参数转换

### 🔧 处理流程

```
原始embedding → Center (减均值) → Whiten (去相关) → L2归一化 → 保存
                   ↑                    ↑
                只用训练集计算       只用训练集计算
```

### 📐 数学原理

**Center（中心化）**：
```python
mean = train_emb.mean(axis=0)
emb_centered = emb - mean
```

**Whiten（白化）**：
```python
# 计算协方差矩阵
cov = (train_centered.T @ train_centered) / n_train

# SVD分解
U, S, _ = np.linalg.svd(cov)

# 白化矩阵
whiten_matrix = U @ diag(1/sqrt(S + 1e-5))

# 应用白化
emb_whitened = emb_centered @ whiten_matrix
```

## 使用方法

### 1. TF-IDF + SVD (Base 特征)

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --title_field title
  # 默认启用 whiten，使用 --no_whiten 关闭
```

**输出文件**：
- `item_text_emb.base.npy` - 处理后的embedding矩阵
- `item_text_emb.base_whiten_stats.npz` - 统计量文件（包含 mean 和 whiten_matrix）

### 2. Qwen3 LLM 特征

```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --dataset Amazon_Beauty \
  --output_mode mean \
  --batch_size 16
  # 默认启用 whiten，使用 --no_whiten 关闭
```

**重要**：必须传 `--dataset` 参数才能只用训练集fit统计量！

**输出文件**：
- `item_text_emb.qwen3.npy` - 处理后的embedding矩阵
- `item_text_emb.qwen3_whiten_stats.npz` - 统计量文件

### 3. 关闭白化功能

如果只想 center + L2归一化（不做白化）：

```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --no_whiten  # 添加此参数
```

## 验证方法

### 检查统计量文件

```python
import numpy as np

# 加载统计量
stats = np.load('dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz')
print("Mean shape:", stats['mean'].shape)
print("Whiten matrix shape:", stats['whiten_matrix'].shape)

# 验证白化后的协方差矩阵接近单位矩阵
emb = np.load('dataset/Amazon_Beauty/item_text_emb.base.npy')
train_emb = emb[1:100]  # 取部分训练集样本
cov = (train_emb.T @ train_emb) / len(train_emb)
print("Covariance diagonal:", np.diag(cov)[:5])  # 应该接近 [1, 1, 1, ...]
print("Covariance off-diagonal max:", np.abs(cov - np.diag(np.diag(cov))).max())  # 应该接近 0
```

### 对比实验指标

运行两个版本对比：

```bash
# 版本1：启用 whiten
python tools/build_item_text_emb_base.py --dataset Amazon_Beauty --output base_with_whiten.npy

# 版本2：不启用 whiten
python tools/build_item_text_emb_base.py --dataset Amazon_Beauty --output base_no_whiten.npy --no_whiten

# 分别训练模型，对比 NDCG@10
```

## 常见问题

### Q1: 为什么在预处理阶段做而不是模型中？

**A**: 
- ✅ 只计算一次，提高效率
- ✅ 保存统计量，推理时复用
- ✅ 避免时间泄漏更容易控制
- ✅ 代码分离清晰

### Q2: whiten_stats.npz 文件有什么用？

**A**: 
- 推理/测试阶段加载该文件，使用**相同的 mean 和 whiten_matrix**
- 确保训练集和测试集使用一致的归一化参数
- 避免测试集信息泄漏到预处理过程

### Q3: 如果忘记传 --dataset 参数会怎样？

**A**: 
- TF-IDF脚本：会报错（必需参数）
- Qwen3脚本：会用**所有数据**fit统计量，存在泄漏风险
- **建议**：务必传 `--dataset` 参数

### Q4: 白化对性能影响多大？

**A**: 
根据辅导要求，白化能：
- 统一 TF-IDF 和 LLM 两路特征的分布
- 消除特征冗余，让交叉网络学得更高效
- 改善数值稳定性
- 预期提升 NDCG@10 约 1-2%（需实验验证）

## 技术细节

### 数值稳定性

- 使用 `float64` 精度计算统计量，避免累积误差
- 白化矩阵添加 `eps=1e-5` 防止除零
- PAD embedding (index=0) 始终保持零向量

### 内存优化

- 只在训练集上计算协方差矩阵（节省内存）
- 统计量保存为 `float32`（兼顾精度和空间）

### 兼容性

- 支持 2D embedding矩阵 (concat/mean 模式)
- 3D 矩阵 (stack 模式) 自动跳过全局白化（需per-view处理）

## 文件修改记录

### build_item_text_emb_base.py
- 新增函数：`_center_whiten_and_normalize()`
- 新增参数：`--no_whiten`
- 默认行为：启用 center + whiten

### build_item_text_emb_qwen3_hf.py
- 新增函数：`_center_whiten_and_normalize()`
- 新增参数：`--no_whiten`
- 修复导入：添加 `Optional` 类型
- 默认行为：启用 center + whiten

## 参考文献

1. PCA Whitening: https://en.wikipedia.org/wiki/Whitening_transformation
2. ZCA Whitening in Deep Learning
3. RecBole论文辅导要求：统一center/whiten + L2/LayerNorm

---

**更新日期**: 2025-12-03
**维护者**: RecBole Team

