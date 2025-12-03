# 最终确认：归一化流程

## ✅ 您的需求已完全满足

> 在text文本离线生成之后只保留center和whiten的效果，L2在sasrec模型侧去做，只做一次

---

## 📊 归一化流程确认

### 文本特征生成（离线）

**脚本：**
- `build_item_text_emb_base.py`
- `build_item_text_emb_qwen3_hf.py`

**处理：**
```python
if enable_whiten:  # True（默认）
    emb = TF-IDF/Qwen3 → SVD → Center → Whiten
    # ✅ 不做L2归一化
    save(emb)
```

**输出文件：**
- ✅ 包含：center + whiten
- ✅ 不包含：L2 normalize
- ✅ 统计量：保存在 `*_whiten_stats.npz`

---

### 模型加载（__init__）

**代码：** `sasrec_align.py` Line 165-174

**处理：**
```python
# 加载特征
emb = load_npy(path)  # center + whiten（无L2）

# L2归一化（唯一1次）
if normalize_text:  # true
    emb = F.normalize(emb, p=2, dim=1)  # ← 唯一的L2归一化

# 冻结
register_buffer(emb)
```

**配置：**
```yaml
# sasrec_align_base.yaml
# sasrec_align_qwen3.yaml  
# sasrec_align_multi_view.yaml
normalize_text: true  # 启用L2归一化
```

**L2归一化次数：1次** ✅

---

### 前向传播（forward）

#### 双路模型

```python
# 从buffer获取
text_raw = self._gather_text_raw(ids)  # 已L2归一化

# 直接使用，无额外归一化 ✅
fusion_input = cat([item_emb, text_raw])
```

**L2归一化次数：0次** ✅

---

#### 多视图模型（改进后）

```python
# 从buffer获取并处理
text_proj = multiview_projection_with_base(...)  # 已L2归一化（的组合）

# ✅ 已移除重复L2归一化（原Line 288-289）
# 直接使用

fusion_input = cat([item_emb, text_proj])
```

**L2归一化次数：0次** ✅

---

## 📋 训练脚本确认

### two_phase_run_tfidf.sh ✅

```bash
# 配置文件
--config_files "sasrec_align_base.yaml"

# 特征路径
item_text_emb_path_base: .../item_text_emb.base.npy  # center+whiten

# 归一化
normalize_text: true  # __init__时L2（1次）
```

**流程：**
```
特征文件[center+whiten] → __init__[L2×1] → forward[L2×0]
```

✅ 符合要求

---

### two_phase_run_tfidf_llm.sh ✅

```bash
# 配置文件
--config_files "sasrec_align_qwen3.yaml"

# 特征路径
item_text_emb_path_base: .../item_text_emb.base.npy    # center+whiten
item_text_emb_path_llm: .../item_text_emb.qwen3.npy    # center+whiten

# 归一化
normalize_text: true  # __init__时L2（1次）
```

**流程：**
```
Base文件[center+whiten] → __init__[L2×1] → forward[L2×0]
LLM文件[center+whiten]  → __init__[L2×1] → forward[L2×0]
```

✅ 符合要求

---

### two_phase_run_multiview_split.sh ✅

```bash
# 配置文件
--config_files "sasrec_align_multi_view.yaml"

# 特征路径
item_text_emb_path_base: .../item_text_emb.base.npy           # center+whiten
item_text_emb_split_dir: .../qwen3_4views/                    # 每个view: center+whiten

# 归一化
normalize_text: true  # __init__时L2（1次）
```

**流程：**
```
Base文件[center+whiten] → __init__[L2×1] → forward[L2×0]
View文件[center+whiten] → __init__[L2×1] → forward[L2×0]
                                           ↓
                               投影+融合（无L2）✅
```

✅ 符合要求（改进后）

---

## ✅ 最终确认

| 检查项 | 状态 | 说明 |
|--------|------|------|
| **特征生成仅做center+whiten** | ✅ | 两个生成脚本都正确 |
| **模型加载做1次L2** | ✅ | __init__时唯一的L2归一化点 |
| **前向传播不做L2** | ✅ | 所有模型都不再归一化 |
| **多视图无重复L2** | ✅ | 已移除Line 288-289 |
| **配置文件一致** | ✅ | 所有配置都用normalize_text: true |

---

## 🎯 设计理念

### 为什么这样设计？

1. **特征文件保留whitening**
   - 离线处理，一次生成多次使用
   - 保留统计量用于推理
   - 特征可复用

2. **模型加载时L2归一化**
   - 统一不同来源特征的scale
   - 简化模型内部处理
   - 便于cosine similarity计算

3. **前向传播不归一化**
   - 避免重复计算
   - 提高训练效率
   - 保持一致性

---

## ⚠️ 关于Whitening的说明

### 当前权衡

**特征文件：** Whitened（Cov=I）  
**模型buffer：** Whitened + L2（Cov≠I）

**说明：**
- L2归一化会改变whitened embeddings的协方差结构
- 原本 `Cov(X_whitened) = I`
- L2后 `Cov(X_L2) ≈ (1/d) × I`

**是否是问题？**
- 理论上：L2破坏了whitening的数学性质
- 实践上：可能仍然有效（需实验验证）

### 可选探索

**实验对比：**
```yaml
# 实验A：当前配置
normalize_text: true  # __init__时L2归一化

# 实验B：纯whitening
normalize_text: false  # 不做L2归一化
```

**对比性能：**
- 观察NDCG@10、Recall@10等指标
- 确定哪种配置更好

**我的预测：**
- normalize_text: true 可能更好（因为cosine_score需要）
- 但需要实验验证

---

## 📚 完整文档

1. **`NORMALIZATION_SUMMARY.md`** - 归一化流程总结（本文档）
2. **`NORMALIZATION_FLOW_ANALYSIS.md`** - 详细流程分析
3. **`NORMALIZATION_VERIFICATION.md`** - 验证方法

---

## 🚀 下一步

### 当前配置可直接使用 ✅

```bash
# 生成特征（center + whiten，无L2）
bash tools/gen_text_emb_beauty_full.sh

# 训练模型（__init__时L2，forward时无L2）
bash two_phase_run_tfidf.sh
bash two_phase_run_tfidf_llm.sh
bash two_phase_run_multiview_split.sh
```

**确认：**
- ✅ L2归一化仅在 __init__ 做1次
- ✅ 所有模型行为一致
- ✅ 符合您的要求

---

**确认日期**: 2025-12-03  
**归一化次数**: 1次（__init__）  
**符合要求**: ✅ 完全符合

