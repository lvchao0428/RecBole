# 归一化流程完整分析

## 🔍 当前归一化流程

### 文本生成阶段（离线）

#### `build_item_text_emb_base.py` 和 `build_item_text_emb_qwen3_hf.py`

```python
# Line 227-255 (两个脚本相同逻辑)

if enable_whiten:  # 默认True
    # Step 1: Center
    emb_centered = emb - mean
    
    # Step 2: Whiten
    emb_whitened = emb_centered @ whiten_matrix
    
    # Step 3: 保存（不做L2）
    return emb_whitened  # ✅ 仅 center + whiten，无L2
    
else:
    # 如果禁用whiten
    emb_centered = emb - mean
    emb_normalized = L2_normalize(emb_centered)  # center + L2
    return emb_normalized
```

**当前状态：**
- ✅ **正确**：whiten时不做L2归一化
- ✅ 保存的文件：仅包含 center + whiten

---

### 模型加载阶段（__init__）

#### `sasrec_align.py` Line 165-174

```python
# 加载特征文件
emb_base = np.load(item_text_emb_path_base)  # [N, 256] (whitened)
emb_llm = np.load(item_text_emb_path_llm)    # [N, 256] (whitened)

# 第一次L2归一化
if self.normalize_text:  # true（所有配置文件）
    with torch.no_grad():
        if emb_base is not None:
            norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
            emb_base = emb_base / norms.clamp_min(1e-8)  # ← L2归一化（第1次）
            
        if emb_llm is not None:
            norms = torch.norm(emb_llm, p=2, dim=1, keepdim=True)
            emb_llm = emb_llm / norms.clamp_min(1e-8)   # ← L2归一化（第1次）

# 注册为buffer
self.register_buffer("item_text_emb_base", emb_base)
self.register_buffer("item_text_emb_llm", emb_llm)
```

**状态：**
- ✅ 在加载时做了**第1次L2归一化**
- ⚠️  这会破坏whitening效果（Cov ≠ I）

---

### 前向传播阶段（forward）

#### SASRecAlign（双路模型）

```python
# _get_fused_item_embeddings 方法

# 1. 获取文本特征（已L2归一化）
text_raw = self._gather_text_raw(all_ids)  # [B, 512]
# text_raw 来自 buffer，已在 __init__ 时做过L2归一化

# 2. 直接使用，无额外归一化
scaled_text = effective_text_weight * text_raw

# 3. 融合
fusion_input = cat([item_emb, scaled_text])  # [B, 768]
```

**状态：**
- ✅ **仅1次L2归一化**（在__init__时）
- ✅ forward时不再归一化

---

#### SASRecAlignMultiView（多视图模型）- 当前

```python
# _fuse_with_cross_network 方法，Line 287-289

# text_raw: 投影后的多视图特征 [B, 512]

# 第二次L2归一化
if self.normalize_text:
    text_raw = F.normalize(text_raw, dim=1)  # ← L2归一化（第2次）❌

# 然后融合
scaled_text = effective_text_weight * text_raw
fusion_input = cat([item_emb, scaled_text])
```

**状态：**
- ❌ **L2归一化做了2次**
  - 第1次：__init__时对原始特征
  - 第2次：forward时对投影后的特征
- ❌ 不一致：双路模型只做1次

---

## 🚨 问题总结

### 问题1：__init__时的L2归一化破坏了whitening

```python
# 特征生成时（正确）：
emb = center + whiten  # Cov(emb) = I ✅

# 模型加载时（有问题）：
if normalize_text:
    emb = L2_normalize(emb)  # Cov(emb) ≠ I ❌
```

**影响：**
- whitening的去相关效果被破坏
- 协方差矩阵不再是单位矩阵

---

### 问题2：多视图模型L2归一化了两次

```python
# __init__时（第1次）
emb_base = L2_normalize(load_npy(base_path))

# forward时（第2次）- 仅多视图模型
text_proj = multiview_projection(...)
if normalize_text:
    text_proj = F.normalize(text_proj)  # 又归一化了一次❌
```

**影响：**
- 冗余操作
- 与双路模型行为不一致

---

## ✅ 改进方案

根据您的需求：
> 在text文本离线生成之后只保留center和whiten的效果，L2在sasrec模型侧去做

### 方案：仅在__init__时做1次L2归一化

**流程：**
```
特征生成（离线）：center + whiten（不做L2）
  ↓ 保存 .npy
模型加载（__init__）：L2 normalize（仅1次）
  ↓ register_buffer
前向传播（forward）：不再归一化
```

---

## 🔧 需要的修改

### 1. 特征生成脚本（已正确）✅

**`build_item_text_emb_base.py` 和 `build_item_text_emb_qwen3_hf.py`：**
- ✅ 当前已正确：whiten时不做L2（Line 245-248注释）
- ✅ 无需修改

---

### 2. 模型代码（需要修改）

#### 修改A：保持__init__的L2归一化

**`sasrec_align.py` Line 165-174：**
- ✅ **保持不变**
- 这是唯一的L2归一化点

#### 修改B：移除forward中的重复L2归一化

**`sasrecalignmultiview.py` Line 287-289：**

```python
# 修改前（错误）：
if self.normalize_text:
    text_raw = F.normalize(text_raw, dim=1)  # 第2次L2归一化❌

# 修改后（正确）：
# 移除这个归一化
# text_raw 已经在 __init__ 时归一化过了
# 投影不会破坏L2归一化（线性变换后再归一化是多余的）
```

---

### 3. 配置文件（保持不变）

**所有 `.yaml` 文件：**
```yaml
normalize_text: true  # 保持
```

这控制 __init__ 时的L2归一化。

---

## 📊 改进后的完整流程

```
┌─────────────────────────────────────────┐
│ 1. 特征生成（离线）                      │
└─────────────────────────────────────────┘
TF-IDF → SVD → center → whiten
  ↓ 保存
item_text_emb.base.npy
  - 仅包含 center + whiten
  - 不包含 L2 normalize
  - Cov(emb) = I（理论上）

┌─────────────────────────────────────────┐
│ 2. 模型初始化（__init__）                │
└─────────────────────────────────────────┘
emb = load_npy(path)
  ↓ if normalize_text
emb = F.normalize(emb, p=2, dim=1)  ← 唯一的L2归一化
  ↓ register_buffer
self.item_text_emb_base = emb
  - 包含 center + whiten + L2
  - Cov(emb) ≈ (1/d) × I （L2后）
  - ||emb|| = 1.0

┌─────────────────────────────────────────┐
│ 3. 前向传播（forward）                   │
└─────────────────────────────────────────┘
text_raw = self.item_text_emb_base[ids]
  ↓ 直接使用（已L2归一化）
text_proj = projection(text_raw)
  ↓ 投影后不再归一化
fusion_input = cat([item_emb, text_proj])
  ↓ 融合
fused_item_emb = fusion_network(fusion_input)
```

**关键点：**
- ✅ L2归一化仅在 __init__ 时做1次
- ✅ forward时不再归一化
- ✅ 简洁高效

---

## 🔍 验证当前状态

让我创建一个验证脚本来检查当前的归一化次数。

