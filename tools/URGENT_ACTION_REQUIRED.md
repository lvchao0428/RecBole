# 🚨 紧急行动：重新生成所有文本特征

## 📋 问题总结

发现并修复了 **SVD + Whitening 顺序问题**，导致白化完全失效：
- 协方差矩阵对角线：0.0039（期望 1.0）
- 白化效果：❌ 失败

**根本原因：** SVD降维后立即做了L2归一化，破坏了白化所需的方差结构。

**详细说明：** [FIX_WHITENING_ORDER.md](./FIX_WHITENING_ORDER.md)

---

## ✅ 立即行动

### Step 1: 删除旧特征（可选备份）

```bash
cd /home/charlie/project/RecBole

# 备份旧特征（可选）
mkdir -p backup/old_embeddings_$(date +%Y%m%d)
mv dataset/Amazon_Beauty/item_text_emb.*.npy backup/old_embeddings_$(date +%Y%m%d)/ 2>/dev/null || true
mv dataset/Amazon_Beauty/qwen3_4views/ backup/old_embeddings_$(date +%Y%m%d)/ 2>/dev/null || true
```

### Step 2: 重新生成所有特征

```bash
# 完整版（推荐）- 生成所有类型特征
bash tools/gen_text_emb_beauty_full.sh
```

**预计耗时：**
- TF-IDF: 1-5 分钟
- Qwen3单视图: 10-30 分钟
- Qwen3多视图: 30-60 分钟
- **总计：约 1-2 小时**

或分步生成：

```bash
# 方案A：仅TF-IDF（快速验证修复）
bash tools/gen_tfidf_only.sh

# 方案B：仅Qwen3多视图
bash tools/gen_qwen3_multiview_only.sh
```

### Step 3: 验证修复效果

```bash
# 批量验证所有特征
bash tools/verify_all_embeddings.sh
```

**期望输出：**
```
📊 协方差矩阵分析:
   - 对角线均值: 1.0023 (期望≈1.0)  ✅
   - 对角线标准差: 0.0456 (期望≈0)   ✅
   - 非对角线最大值: 0.1234 (期望≈0) ✅
✅ 白化效果良好
```

### Step 4: 重新运行实验

```bash
# TF-IDF基线实验
bash two_phase_run_tfidf.sh

# Qwen3多视图实验
bash two_phase_run_multiview_split.sh
```

---

## 🔍 修改的文件

### 已修复的脚本

✅ `tools/build_item_text_emb_base.py`
- 移除 `_fit_tfidf_svd` 中的 post-SVD L2归一化
- 移除 `_fit_on_train_transform_all` 中的 post-SVD L2归一化

✅ `tools/build_item_text_emb_qwen3_hf.py`
- `_apply_truncated_svd` 添加 `normalize` 参数
- Per-view SVD 使用 `normalize=False`
- Final SVD 使用 `normalize=True`

### 新增/更新的文档

📄 `FIX_WHITENING_ORDER.md` - 详细的问题分析和修复说明  
📄 `TEXT_EMB_GENERATION_GUIDE.md` - 更新了验证章节  
📄 `VERIFICATION_GUIDE.md` - 验证工具使用指南

---

## 📊 预期改进

### 白化质量

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 对角线均值 | 0.0039 ❌ | ~1.00 ✅ |
| 非对角线均值 | 0.0004 | ~0.01 ✅ |
| 白化效果 | 失败 ❌ | 良好 ✅ |

### 模型性能

白化的作用：
- ✅ 去除特征间的相关性（decorrelation）
- ✅ 标准化特征scale（每个维度方差=1）
- ✅ 提高模型训练稳定性
- ✅ 可能提升推荐性能指标（NDCG, Recall等）

---

## ⚠️ 注意事项

### 1. 不要跳过验证步骤

```bash
# ❌ 错误：直接运行实验
bash two_phase_run_tfidf.sh

# ✅ 正确：先验证，再实验
bash tools/verify_all_embeddings.sh
bash two_phase_run_tfidf.sh
```

### 2. 保留旧特征备份（可选）

如果需要对比新旧特征的实验结果：
```bash
# 重命名而非删除
mv dataset/Amazon_Beauty/item_text_emb.base.npy \
   dataset/Amazon_Beauty/item_text_emb.base.old.npy
```

### 3. 检查磁盘空间

```bash
# 检查可用空间
df -h dataset/

# 预计需要空间（Amazon_Beauty）
# - TF-IDF: ~130 MB
# - Qwen3单视图: ~300 MB
# - Qwen3多视图: ~150 MB (分视图) + ~130 MB (拼接)
# 总计: ~700 MB
```

---

## 🐛 如果遇到问题

### Q1: 验证仍然失败

参考 [FIX_WHITENING_ORDER.md](./FIX_WHITENING_ORDER.md) 的调试章节。

### Q2: 生成过程中OOM

```bash
# 减小batch_size
python tools/build_item_text_emb_qwen3_hf.py \
  --batch_size 8 \  # 默认16
  ...
```

### Q3: 特征文件损坏

```bash
# 删除并重新生成
rm dataset/Amazon_Beauty/item_text_emb.base.npy
bash tools/gen_tfidf_only.sh
```

---

## 📞 获取帮助

如果遇到其他问题：

1. 查看详细文档：[FIX_WHITENING_ORDER.md](./FIX_WHITENING_ORDER.md)
2. 检查验证输出：`bash tools/verify_all_embeddings.sh`
3. 检查生成日志：查看终端输出

---

## ✅ 完成检查清单

- [ ] 备份/删除旧特征文件
- [ ] 重新生成所有特征
- [ ] 验证白化效果（协方差矩阵对角线 ≈ 1.0）
- [ ] 重新运行实验
- [ ] 对比新旧实验结果（可选）
- [ ] 更新实验记录

---

**创建日期**: 2025-12-03  
**优先级**: 🔴 URGENT  
**预计耗时**: 1-2 小时（生成特征）  
**影响**: 所有使用文本特征的实验

