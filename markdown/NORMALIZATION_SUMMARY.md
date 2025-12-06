# 归一化流程总结 - 最终确认

## ✅ 您的需求

> 在text文本离线生成之后只保留center和whiten的效果，L2在sasrec模型侧去做，只做一次

## ✅ 当前实现状态

### 完全符合要求！

| 阶段 | 操作 | 位置 | 次数 |
|------|------|------|------|
| **特征生成（离线）** | center + whiten | `build_item_text_emb_*.py` | ✅ 1次 |
| **模型加载（__init__）** | L2 normalize | `sasrec_align.py` Line 165-174 | ✅ 1次 |
| **前向传播（forward）** | 无归一化 | 所有模型 | ✅ 0次 |

**L2归一化总次数：1次（仅在模型加载时）** ✅

---

## 📋 详细验证

### 1. 特征生成脚本 ✅

#### `build_item_text_emb_base.py`

```python
# Line 227-255
if enable_whiten:  # 默认True
    # Center + Whiten
    emb_centered = emb - mean
    emb_whitened = emb_centered @ whiten_matrix
    
    # ✅ 不做L2归一化（Line 245-248注释明确说明）
    return emb_whitened
else:
    # 仅当未启用whiten时才做L2
    return L2_normalize(emb_centered)
```

#### `build_item_text_emb_qwen3_hf.py`

```python
# Line 307-333
if enable_whiten:  # 默认True
    # Center + Whiten
    emb_centered = emb - mean
    emb_whitened = emb_centered @ whiten_matrix
    
    # ✅ 不做L2归一化（Line 324-326注释明确说明）
    return emb_whitened
else:
    # 仅当未启用whiten时才做L2
    return L2_normalize(emb_centered)
```

**输出文件内容：** center + whiten（无L2）✅

---

### 2. 模型加载 ✅

#### `sasrec_align.py` Line 165-174

```python
# 加载特征文件
emb_base = np.load(base_path)  # center + whiten（无L2）
emb_llm = np.load(llm_path)    # center + whiten（无L2）

# 在__init__时做L2归一化（唯一1次）
if self.normalize_text:  # true（所有配置）
    with torch.no_grad():
        emb_base = F.normalize(emb_base, p=2, dim=1)  # ← L2归一化（唯一）
        emb_llm = F.normalize(emb_llm, p=2, dim=1)    # ← L2归一化（唯一）

# 冻结为buffer
self.register_buffer("item_text_emb_base", emb_base)  # center + whiten + L2
self.register_buffer("item_text_emb_llm", emb_llm)
```

**L2归一化次数：1次** ✅

---

### 3. 前向传播 ✅

#### 双路模型（SASRecAlign）

```python
# _get_fused_item_embeddings 方法

# 从buffer获取（已L2归一化）
text_raw = self._gather_text_raw(all_ids)
# ✅ 不再归一化

scaled_text = effective_weight * text_raw
fusion_input = cat([item_emb, scaled_text])
fused = fusion_network(fusion_input)
```

**L2归一化次数：0次** ✅

---

#### 多视图模型（SASRecAlignMultiView）- 改进后

```python
# _get_fused_item_embeddings 方法

# 从buffer获取（已L2归一化）
view_stack = self._gather_text_views(all_ids)

# Gate加权 + 拼接base + 投影
text_proj = multiview_projection_with_base(view_stack, base_feat)

# ✅ 不再归一化（已移除Line 288-289的重复L2）

scaled_text = effective_weight * text_proj
fusion_input = cat([item_emb, scaled_text])
fused = fusion_network(fusion_input)
```

**L2归一化次数：0次** ✅

---

## 🔧 改动总结

### 已完成的修改

1. **特征生成脚本（之前已修复）**
   - ✅ `build_item_text_emb_base.py`：whiten时不做L2
   - ✅ `build_item_text_emb_qwen3_hf.py`：whiten时不做L2

2. **多视图模型（本次修复）**
   - ✅ `sasrecalignmultiview.py` Line 287-293：移除重复L2归一化

3. **双路模型（无需修改）**
   - ✅ `sasrec_align.py`：本身就只做1次L2

---

## 📊 配置文件

### 所有相关配置

```yaml
# sasrec_align_base.yaml
# sasrec_align_qwen3.yaml
# sasrec_align_multi_view.yaml

normalize_text: true  # 控制__init__时的L2归一化
```

**保持不变** ✅

---

## 🎯 归一化策略总结

### 完整的归一化链

```
离线特征生成:
  ├─ TF-IDF vectorization
  ├─ (Optional) Pre-SVD L2（数值稳定性）
  ├─ SVD降维（不归一化）
  ├─ Center（中心化）
  ├─ Whiten（白化）
  └─ 保存（不做L2）✅

模型加载(__init__):
  ├─ Load .npy
  ├─ L2 normalize（唯一1次）✅
  └─ Register buffer

前向传播(forward):
  ├─ Gather from buffer
  ├─ Projection
  ├─ Fusion
  └─ 无归一化 ✅

打分(predict):
  ├─ Get seq_output
  ├─ Get fused_item_emb
  └─ (Optional) L2 normalize（仅用于cosine score）
```

**注意：** 打分时的L2归一化是针对fused_item_emb（融合后的特征），不是原始文本特征

---

## 🔍 验证方法

### 方法1：代码审查 ✅

检查以下位置是否有L2归一化：
- [x] 特征生成：whiten时无L2 ✅
- [x] 模型__init__：有L2（1次）✅
- [x] 双路forward：无L2 ✅
- [x] 多视图forward：无L2（已移除）✅

---

### 方法2：添加调试输出

```python
# 在 sasrec_align.py __init__ 中添加
if self.normalize_text:
    print(f"[DEBUG] L2 normalizing text features at __init__")
    print(f"  Base before: mean_norm={torch.norm(emb_base[1:100], p=2, dim=1).mean():.4f}")
    emb_base = F.normalize(emb_base, p=2, dim=1)
    print(f"  Base after: mean_norm={torch.norm(emb_base[1:100], p=2, dim=1).mean():.4f}")

# 在 sasrecalignmultiview.py _fuse_with_cross_network 中添加
print(f"[DEBUG] text_raw at fusion: mean_norm={torch.norm(text_raw[:10], p=2, dim=1).mean():.4f}")
# 如果已L2归一化，应该约等于1.0
```

**期望输出：**
```
[DEBUG] L2 normalizing text features at __init__
  Base before: mean_norm=1.4142  (whitened，约sqrt(2))
  Base after: mean_norm=1.0000  (L2归一化后)
  
[DEBUG] text_raw at fusion: mean_norm=0.9998  (投影后，仍接近1.0)
```

---

### 方法3：运行验证脚本

```bash
# 创建一个验证脚本
python -c "
import torch
import numpy as np

# 1. 检查特征文件（应该是whitened，不是L2归一化的）
emb = np.load('dataset/Amazon_Beauty/item_text_emb.base.npy')
norms = np.linalg.norm(emb[1:100], axis=1)
print(f'File norms: mean={norms.mean():.4f} (expect ~sqrt(d))')

# 2. 检查统计量文件
stats = np.load('dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz')
print(f'Stats keys: {list(stats.keys())}')
print(f'Has whiten_matrix: {\"whiten_matrix\" in stats}')
"
```

**期望输出：**
```
File norms: mean=1.4142 (expect ~sqrt(d))  ← whitened，不是L2
Stats keys: ['mean', 'whiten_matrix']
Has whiten_matrix: True
```

---

## 📚 相关文档

- **完整分析：** `NORMALIZATION_FLOW_ANALYSIS.md`
- **归一化验证：** `NORMALIZATION_VERIFICATION.md`
- **多视图改进：** `MULTIVIEW_IMPROVEMENT_README.md`

---

## 🎉 总结

### 最终确认

✅ **特征生成脚本：**
- `build_item_text_emb_base.py`：仅 center + whiten
- `build_item_text_emb_qwen3_hf.py`：仅 center + whiten

✅ **模型代码：**
- 所有模型在 __init__ 时做 L2 normalize（唯一1次）
- forward 时不再归一化

✅ **配置文件：**
- `normalize_text: true`（所有配置）
- 控制 __init__ 的 L2 归一化

✅ **完全符合您的要求！**

---

**验证日期**: 2025-12-03  
**状态**: ✅ 确认正确  
**L2归一化次数**: 1次（仅__init__）

