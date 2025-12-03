# 归一化流程验证与确认

## ✅ 确认：仅在模型加载时做1次L2归一化

根据您的需求：
> 在text文本离线生成之后只保留center和whiten的效果，L2在sasrec模型侧去做，只做一次

---

## 📊 完整归一化流程

### 阶段1：特征生成（离线）

**脚本：**
- `build_item_text_emb_base.py`
- `build_item_text_emb_qwen3_hf.py`

**处理流程：**
```python
# Step 1: TF-IDF / Qwen3 embedding
raw_emb = generate_embeddings(texts)

# Step 2: SVD降维（如果需要）
reduced_emb = TruncatedSVD.transform(raw_emb)
# 不做L2归一化（保留方差结构）

# Step 3: Center（中心化）
emb_centered = reduced_emb - mean(train_emb)

# Step 4: Whiten（白化）
if enable_whiten:  # 默认True
    emb_whitened = emb_centered @ whiten_matrix
    emb_final = emb_whitened  # ← 不做L2归一化
else:
    emb_final = L2_normalize(emb_centered)  # 仅当未whiten时

# Step 5: 保存
np.save(output_path, emb_final)
```

**输出：**
- ✅ 文件包含：center + whiten
- ✅ 文件不包含：L2 normalize
- ✅ 特性：`Cov(emb) = I`（单位协方差矩阵）

---

### 阶段2：模型加载（__init__）

**代码位置：** `sasrec_align.py` Line 165-174

**处理流程：**
```python
# Step 1: 加载特征文件
emb_base = load_npy(base_path)  # [N, 256] center+whiten
emb_llm = load_npy(llm_path)    # [N, 256] center+whiten

# Step 2: L2归一化（唯一一次）
if self.normalize_text:  # true
    emb_base = F.normalize(emb_base, p=2, dim=1)  # ← 唯一的L2归一化
    emb_llm = F.normalize(emb_llm, p=2, dim=1)    # ← 唯一的L2归一化

# Step 3: 冻结为buffer
self.register_buffer("item_text_emb_base", emb_base)
self.register_buffer("item_text_emb_llm", emb_llm)
```

**输出：**
- ✅ Buffer包含：center + whiten + L2
- ✅ 特性：`||emb|| = 1.0`（单位范数）
- ⚠️  注意：`Cov(emb) ≠ I`（L2破坏了白化）

---

### 阶段3：前向传播（forward）

**代码位置：** `sasrec_align.py` Line 796-862

**处理流程：**

#### 双路模型（SASRecAlign）

```python
# Step 1: 获取文本特征（已L2归一化）
text_raw = self._gather_text_raw(all_ids)  # [B, 512]
# text_raw 来自 buffer，已L2归一化

# Step 2: 投影（Cross + Deep + Linear）
# 无额外归一化

# Step 3: 融合
scaled_text = effective_weight * text_raw
fusion_input = cat([item_emb, scaled_text])
fused = fusion_network(fusion_input)
```

**归一化次数：0次**（已在__init__时做过）

---

#### 多视图模型（SASRecAlignMultiView）- 改进前

```python
# Step 1: 获取多视图特征（已L2归一化）
view_stack = self._gather_text_views(all_ids)  # [B, 4, 256]
# view embeddings 来自 buffer，已L2归一化

# Step 2: Gate加权 + 投影
text_concat = gate_weighted_concat(view_stack)  # [B, 1024]
text_proj = Linear(text_concat)  # [B, 512]

# Step 3: 又做了一次L2归一化（问题所在）❌
if self.normalize_text:
    text_proj = F.normalize(text_proj, dim=1)  # 第2次L2归一化

# Step 4: 融合
scaled_text = effective_weight * text_proj
fusion_input = cat([item_emb, scaled_text])
```

**归一化次数：2次**（__init__ 1次 + forward 1次）❌

---

#### 多视图模型（SASRecAlignMultiView）- 改进后

```python
# Step 1: 获取多视图特征（已L2归一化）
view_stack = self._gather_text_views(all_ids)

# Step 2: Gate加权 + 拼接base + 投影
text_proj = multiview_projection_with_base(...)  # [B, 512]

# Step 3: 直接使用（不再归一化）✅
# 已移除重复的L2归一化

# Step 4: 融合
scaled_text = effective_weight * text_proj
fusion_input = cat([item_emb, scaled_text])
```

**归一化次数：1次**（仅__init__）✅

---

## 📋 修改清单

### ✅ 已完成的修改

1. **特征生成脚本**
   - ✅ `build_item_text_emb_base.py`：whiten时不做L2（Line 245-248）
   - ✅ `build_item_text_emb_qwen3_hf.py`：whiten时不做L2（Line 324-326）

2. **多视图模型代码**
   - ✅ `sasrecalignmultiview.py`：移除forward中的重复L2（Line 287-293）

### ⚠️ 需要注意的设计选择

**当前设计（__init__时L2归一化）的影响：**

```python
# 特征生成时：
emb = center + whiten  # Cov(emb) = I ✅

# 模型加载时：
emb = L2_normalize(emb)  # Cov(emb) ≠ I ❌
```

**问题：** L2归一化会破坏whitening的协方差性质

**影响：**
- whitening的去相关效果部分丧失
- 但仍保留了中心化和部分去相关

---

## 🤔 设计讨论

### 选项A：当前方案（在__init__做L2）

**流程：**
```
特征生成：center + whiten（不做L2）
模型加载：L2 normalize（1次）
前向传播：不归一化
```

**优点：**
- ✅ L2归一化仅做1次
- ✅ 代码改动最小
- ✅ 特征文件保留whitening统计量

**缺点：**
- ❌ L2归一化破坏了whitening的Cov=I性质
- ❌ whitening的优势打折扣

---

### 选项B：完全不做L2归一化（纯whitening）

**流程：**
```
特征生成：center + whiten（不做L2）
模型加载：不归一化
前向传播：不归一化
```

**实施：**
```yaml
# 所有配置文件
normalize_text: false  # 关闭L2归一化
```

**优点：**
- ✅ 完全保留whitening效果（Cov=I）
- ✅ 理论上最优

**缺点：**
- ⚠️  需要模型适配非L2归一化的特征
- ⚠️  可能影响cosine_score的效果
- ⚠️  需要实验验证性能

---

### 选项C：在需要时动态归一化

**流程：**
```
特征生成：center + whiten（不做L2）
模型加载：不归一化
前向传播：在需要的地方归一化（如cosine_score）
```

**实施：**
```python
# __init__
emb = load_npy(path)  # 不归一化
self.register_buffer("item_text_emb_base", emb)

# forward (仅在需要时)
if self.cosine_score:
    # 仅在打分时归一化
    seq_n = F.normalize(seq_output, dim=1)
    item_n = F.normalize(fused_item_emb, dim=1)
    score = cosine_scale * (seq_n · item_n)
```

**优点：**
- ✅ 保留whitening效果
- ✅ 按需归一化
- ✅ 灵活性高

**缺点：**
- ⚠️  代码改动较大
- ⚠️  需要仔细处理各个使用点

---

## 🎯 推荐方案

### 方案1：当前方案（短期）✅

**保持现状：**
- 特征生成：center + whiten（不做L2）
- 模型加载：L2 normalize（1次）
- 前向传播：不归一化

**适用于：**
- 快速验证
- 最小改动
- 与现有代码兼容

---

### 方案2：纯whitening（长期探索）

**试验配置：**
```yaml
normalize_text: false  # 关闭L2归一化
```

**对比实验：**
- 实验A：normalize_text=true（当前）
- 实验B：normalize_text=false（纯whitening）
- 对比性能差异

**如果方案2更好：**
- 采用纯whitening
- 更新所有配置文件

---

## 📝 当前状态确认

### 文本生成脚本 ✅

```python
# build_item_text_emb_base.py
# build_item_text_emb_qwen3_hf.py

if enable_whiten:  # True（默认）
    emb = center + whiten  # 不做L2 ✅
else:
    emb = center + L2
```

**确认：** ✅ 正确

---

### 模型加载（双路）✅

```python
# sasrec_align.py Line 165-174

if normalize_text:  # True（所有配置）
    emb_base = F.normalize(emb_base, p=2, dim=1)  # L2归一化（唯一1次）✅
    emb_llm = F.normalize(emb_llm, p=2, dim=1)
```

**确认：** ✅ 仅1次L2归一化

---

### 前向传播（双路）✅

```python
# sasrec_align.py _get_fused_item_embeddings

text_raw = self._gather_text_raw(all_ids)  # 来自buffer，已L2归一化
# 无额外归一化 ✅
scaled_text = effective_weight * text_raw
fusion_input = cat([item_emb, scaled_text])
```

**确认：** ✅ 不再归一化

---

### 前向传播（多视图）- 改进前 ❌

```python
# sasrecalignmultiview.py Line 288-289（改进前）

text_proj = multiview_projection(...)
if self.normalize_text:
    text_proj = F.normalize(text_proj, dim=1)  # 第2次L2 ❌
```

**确认：** ❌ 重复归一化

---

### 前向传播（多视图）- 改进后 ✅

```python
# sasrecalignmultiview.py Line 287-293（改进后）

text_proj = multiview_projection(...)
# 不再归一化 ✅
# 已移除重复的L2归一化
```

**确认：** ✅ 仅__init__时归一化

---

## 📊 完整流程图

```
┌──────────────────────────────────────────────────────────┐
│           特征生成（离线，一次性）                          │
├──────────────────────────────────────────────────────────┤
│ TF-IDF/Qwen3 → SVD → Center → Whiten                    │
│                                ↓                         │
│                       保存为 .npy 文件                    │
│                         (不含L2)                          │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│         模型初始化（每次训练，运行一次）                     │
├──────────────────────────────────────────────────────────┤
│ 加载 .npy → L2 Normalize → Register Buffer              │
│              ↑                                           │
│         唯一的L2归一化点                                   │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│      前向传播（每个batch，运行N次）                         │
├──────────────────────────────────────────────────────────┤
│ 从Buffer读取 → 投影 → 融合 → 输出                         │
│ (已L2归一化)                                              │
│                                                          │
│ ✅ 不再做L2归一化                                         │
└──────────────────────────────────────────────────────────┘
```

**L2归一化次数：1次（仅在__init__）** ✅

---

## 🔧 关键代码位置

### 1. 特征生成（不做L2）

**`build_item_text_emb_base.py` Line 245-248：**
```python
# NOTE: Do NOT L2 normalize after whitening!
# Whitening already decorrelates features and sets Cov(X) = I
# L2 normalization would destroy this property
# If the model needs L2-normalized embeddings, do it at model load time
```

**`build_item_text_emb_qwen3_hf.py` Line 324-326：**
```python
# NOTE: Do NOT L2 normalize after whitening!
# Whitening already decorrelates features and sets Cov(X) = I
# L2 normalization would destroy this property
```

---

### 2. 模型加载（唯一L2）

**`sasrec_align.py` Line 165-174：**
```python
if self.normalize_text:  # true
    with torch.no_grad():
        if emb_base is not None:
            emb_base = emb_base / torch.norm(emb_base, p=2, dim=1, keepdim=True).clamp_min(1e-8)
        if emb_llm is not None:
            emb_llm = emb_llm / torch.norm(emb_llm, p=2, dim=1, keepdim=True).clamp_min(1e-8)
```

---

### 3. 前向传播（不归一化）

**双路模型 `sasrec_align.py`：**
- ✅ 无L2归一化
- ✅ 直接使用buffer中的特征

**多视图模型 `sasrecalignmultiview.py`（改进后）：**
- ✅ 移除了Line 288-289的重复L2归一化
- ✅ 直接使用投影后的特征

---

## 📊 配置文件确认

### 所有训练脚本的配置

**`sasrec_align_base.yaml` (two_phase_run_tfidf.sh)：**
```yaml
normalize_text: true  # ✅ 在__init__时L2归一化
```

**`sasrec_align_qwen3.yaml` (two_phase_run_tfidf_llm.sh)：**
```yaml
normalize_text: true  # ✅ 在__init__时L2归一化
```

**`sasrec_align_multi_view.yaml` (two_phase_run_multiview_split.sh)：**
```yaml
normalize_text: true  # ✅ 在__init__时L2归一化
```

**确认：** ✅ 所有配置一致

---

## ✅ 最终确认

### 归一化流程总结

| 阶段 | 操作 | 次数 | 位置 |
|------|------|------|------|
| **特征生成** | center + whiten | 1次（离线） | `build_item_text_emb_*.py` |
| **模型加载** | L2 normalize | 1次（__init__） | `sasrec_align.py` Line 165-174 |
| **前向传播** | 无归一化 | 0次 | 所有模型 |

**L2归一化总次数：1次** ✅

---

### 双路模型流程 ✅

```
离线生成：
  TF-IDF → center + whiten → save [无L2]
  Qwen3 → center + whiten → save [无L2]
  
__init__:
  load → L2 normalize (1次) → buffer
  
forward:
  gather → concat → project → fuse [无L2]
```

**L2次数：1** ✅

---

### 多视图模型流程 ✅

```
离线生成：
  TF-IDF → center + whiten → save [无L2]
  4×Qwen3 → center + whiten → save [无L2]
  
__init__:
  load base → L2 normalize (1次) → buffer
  load 4 views → L2 normalize (1次) → buffer
  
forward:
  gather views → SENet → gate → concat base → 
  project → fuse [无L2]  ← 已移除重复L2 ✅
```

**L2次数：1** ✅

---

## 🎯 结论

### ✅ 符合您的要求

> 在text文本离线生成之后只保留center和whiten的效果，L2在sasrec模型侧去做，只做一次

**当前实现：**
1. ✅ 特征生成：仅 center + whiten（不做L2）
2. ✅ 模型加载：L2 normalize（唯一1次）
3. ✅ 前向传播：不再归一化

**所有模型一致：**
- ✅ 双路模型（SASRecAlign）
- ✅ 多视图模型（SASRecAlignMultiView）- 改进后
- ✅ 单路模型（TF-IDF only / Qwen3 only）

---

## ⚠️ 关于Whitening的讨论

### 当前的权衡

**特征文件：** whitened（Cov=I）  
**模型buffer：** whitened + L2（Cov≠I）

**问题：** L2归一化破坏了whitening的协方差性质

### 可选改进

如果希望完全保留whitening效果，可以尝试：

```yaml
# 实验配置
normalize_text: false  # 关闭L2归一化
```

**然后对比性能：**
- 实验A：`normalize_text: true`（当前）
- 实验B：`normalize_text: false`（纯whitening）

观察哪个性能更好。

---

**文档日期**: 2025-12-03  
**确认状态**: ✅ L2归一化仅做1次（在__init__）  
**符合要求**: ✅ 是

