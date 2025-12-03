# 文本特征验证指南

验证生成的文本特征是否正确应用了 center + whiten 预处理。

## 🔍 验证工具

### 1. 单文件验证 - `verify_whiten.py`

验证单个或多个embedding文件的白化效果。

#### 基本用法

```bash
# 验证TF-IDF特征
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# 验证Qwen3单视图
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy

# 验证Qwen3多视图
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy

# 验证单个视图
python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_0.npy
```

#### 批量验证

```bash
# 验证多个文件
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.base.npy \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
```

---

### 2. 批量验证脚本 - `verify_all_embeddings.sh`

自动发现并验证所有生成的embedding文件。

```bash
# 运行批量验证
bash tools/verify_all_embeddings.sh
```

此脚本会自动查找以下文件：
- `item_text_emb.base.npy` (TF-IDF)
- `item_text_emb.qwen3.base.npy` (Qwen3单视图)
- `item_text_emb.qwen3.multiview.npy` (Qwen3多视图拼接)
- `qwen3_4views/view_*.npy` (Qwen3分视图)

---

## 📊 验证指标说明

### ✅ 必须满足的条件

1. **PAD向量为零**
   - `norm(emb[0]) < 1e-6`
   - PAD行（第0行）应为全零向量

2. **L2归一化**
   - 所有非PAD向量的范数应接近1.0
   - `mean(||emb[i]||) ≈ 1.0, std ≈ 0`

3. **中心化**
   - 均值向量接近零
   - `mean(abs(mean_vec)) < 0.1`

4. **白化（如果启用）**
   - 协方差矩阵对角线接近1.0
   - 非对角线元素接近0
   - `diag(cov) ≈ 1.0, off_diag(cov) ≈ 0`

---

## 📝 输出示例

### 成功案例

```
============================================================
验证 Center + Whiten 预处理
============================================================

✅ 加载embedding: item_text_emb.base.npy
   - 形状: (12102, 256)
   - 数据类型: float16
   - 文件大小: 6.01 MB
✅ 找到统计量文件: dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz
   - mean shape: (1, 256)
   - whiten_matrix shape: (256, 256)
✅ PAD embedding 为零向量 (norm=0.00e+00)
✅ Embedding已L2归一化 (mean=0.9998, std=0.0012)
✅ Embedding已中心化 (mean_abs=0.0234)

📊 协方差矩阵分析 (sample_size=500):
   - 对角线均值: 1.0023 (期望≈1.0)
   - 对角线标准差: 0.0456 (期望≈0)
   - 非对角线最大值: 0.1234 (期望≈0)
   - 非对角线平均值: 0.0089 (期望≈0)
✅ 白化效果良好

✅ 当前文件验证通过
```

### 未启用白化的情况

```
⚠️  未找到统计量文件: dataset/Amazon_Beauty/item_text_emb.old.npy_whiten_stats.npz
✅ PAD embedding 为零向量 (norm=0.00e+00)
✅ Embedding已L2归一化 (mean=0.9999, std=0.0008)
⚠️  Embedding未充分中心化 (mean_abs=0.3456)

📊 协方差矩阵分析 (sample_size=500):
   - 对角线均值: 1.2345 (期望≈1.0)
   - 对角线标准差: 0.4567 (期望≈0)
   - 非对角线最大值: 0.8901 (期望≈0)
   - 非对角线平均值: 0.1234 (期望≈0)
❌ 未检测到白化效果

❌ 当前文件验证失败
```

---

## 🔧 路径支持

脚本现已支持：

### ✅ 相对路径（推荐）

从项目根目录出发的相对路径：
```bash
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
```

### ✅ 绝对路径

完整的系统路径：
```bash
python tools/verify_whiten.py /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy
```

### ✅ 自动路径解析

脚本会自动：
1. 尝试相对于项目根目录解析路径
2. 如果文件不存在，尝试作为绝对路径
3. 提供清晰的错误提示

---

## 🐛 常见问题

### Q1: 文件不存在错误

```
❌ 文件不存在: dataset/Amazon_Beauty/item_text_emb.base.npy
   提示: 请检查路径是否正确（支持相对路径和绝对路径）
```

**解决方案：**
1. 确认文件已生成：`ls -lh dataset/Amazon_Beauty/item_text_emb.*.npy`
2. 检查路径拼写是否正确
3. 确保从项目根目录运行脚本

### Q2: 白化效果一般

```
⚠️  白化效果一般（可能样本量不足或未启用whiten）
```

**可能原因：**
1. 使用了 `--no_whiten` 参数（只有center，无whiten）
2. 训练集样本量较小
3. 数据本身方差较小

**解决方案：**
- 如果是有意禁用白化，可以忽略此警告
- 否则重新生成特征，确保未使用 `--no_whiten`

### Q3: 统计量文件未找到

```
⚠️  未找到统计量文件: dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz
```

**原因：**
- 使用旧版生成脚本（不保存统计量）
- 生成过程中出错

**解决方案：**
- 使用新版脚本重新生成：`bash tools/gen_text_emb_beauty_full.sh`

---

## 📦 验证流程建议

### 标准验证流程

```bash
# 1. 生成所有特征
bash tools/gen_text_emb_beauty_full.sh

# 2. 批量验证
bash tools/verify_all_embeddings.sh

# 3. 检查验证报告
# 如果所有文件都通过验证，即可开始训练实验
```

### 针对性验证

```bash
# 只验证TF-IDF
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# 只验证Qwen3多视图
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy

# 验证所有分视图
python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_{0,1,2,3}.npy
```

---

## 🎯 验证清单

生成特征后，请确认：

- [ ] PAD向量为零向量
- [ ] 所有非PAD向量已L2归一化
- [ ] 嵌入已中心化（如果启用whiten）
- [ ] 协方差矩阵接近单位矩阵（如果启用whiten）
- [ ] 统计量文件已保存（`*_whiten_stats.npz`）
- [ ] 文件大小合理（与预期维度匹配）

---

**更新日期**: 2025-12-03  
**版本**: v2.0 (支持批量验证和路径自动解析)

