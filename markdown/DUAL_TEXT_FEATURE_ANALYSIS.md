# 双路文本特征处理分析

## 📋 需求描述

> 文本侧标准化+对齐做扎实：两路（TF‑IDF/SVD-->256、Qwen3-->线性-->256）统一center/whiten + L2/LayerNorm，再进同一个交叉层。

## 🔍 当前实现分析：`two_phase_run_tfidf_llm.sh`

### 配置文件：`sasrec_align_qwen3.yaml`

```yaml
# 双路文本特征路径
item_text_emb_path_base: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy  # TF-IDF路径
item_text_emb_path_llm: dataset/Amazon_Beauty/item_text_emb.qwen3.npy  # Qwen3路径

# 关键配置
use_llm: true           # 启用LLM
use_align: true         # 启用对齐
use_cross: true         # 启用交叉网络
text_cross_layer_num: 2 # 交叉层数量

# 归一化配置
normalize_text: true    # 文本L2归一化
text_proj_norm: true    # 投影后LayerNorm
fused_item_norm: true   # 融合后LayerNorm
```

---

## 📐 当前处理流程（代码分析）

### Step 1: 加载特征（`__init__` 方法，165-177行）

```python
# 1.1 加载两路特征
emb_base = self._load_text_embeddings(item_text_emb_path_base, self.n_items)  # [N, 256]
emb_llm = self._load_text_embeddings(item_text_emb_path_llm, self.n_items)    # [N, 256]

# 1.2 L2归一化（如果 normalize_text=true）
if self.normalize_text:
    with torch.no_grad():
        if emb_base is not None:
            norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
            emb_base = emb_base / norms.clamp_min(1e-8)  # L2归一化
            
        if emb_llm is not None:
            norms = torch.norm(emb_llm, p=2, dim=1, keepdim=True)
            emb_llm = emb_llm / norms.clamp_min(1e-8)   # L2归一化

# 1.3 注册为buffer（冻结，不参与梯度）
self.register_buffer("item_text_emb_base", emb_base)
self.register_buffer("item_text_emb_llm", emb_llm)
```

**当前状态：**
- ✅ 两路特征都做了 L2 归一化
- ❌ 没有 center（中心化）
- ❌ 没有 whiten（白化）

---

### Step 2: 拼接特征（`_gather_text_raw` 方法，688-696行）

```python
def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
    parts = []
    if self._text_mode in ("base", "both") and self.item_text_emb_base is not None:
        parts.append(self.item_text_emb_base[ids_flat])  # [B, 256]
    if self._text_mode in ("llm", "both") and self.item_text_emb_llm is not None:
        parts.append(self.item_text_emb_llm[ids_flat])   # [B, 256]
    
    # 拼接：[B, 256] + [B, 256] -> [B, 512]
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
```

**当前状态：**
- ✅ 两路特征拼接成 512 维
- ❌ 拼接前没有额外的归一化

---

### Step 3: 投影到hidden_size（`_project_text` 方法，698-718行）

```python
def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
    # raw: [B, 512] (拼接后)
    
    if self.use_cross:
        # 3.1 SENet增强（可选）
        if self.text_amplifier is not None:
            raw = self.text_amplifier(raw)  # [B, 512]
        
        # 3.2 DCN Cross Network
        cross_out = self.text_cross(raw)     # [B, 512]
        cross_out = self.text_cross_dropout(cross_out)
        
        # 3.3 Deep Network
        deep_out = self.text_deep(raw)       # [B, hidden_size]
        
        # 3.4 拼接 + 投影
        fused = torch.cat([cross_out, deep_out], dim=1)  # [B, 512+256]
        proj = self.text_predictor(fused)   # [B, 256] 投影到hidden_size
    else:
        # 简单线性投影
        proj = self.item_text_proj(raw)      # [B, 512] -> [B, 256]
    
    # 3.5 LayerNorm归一化（如果配置）
    if self.text_proj_norm is not None:
        proj = self.text_proj_norm(proj)     # LayerNorm([B, 256])
    
    return proj  # [B, 256]
```

**当前状态：**
- ✅ 通过 Cross Network + Deep Network 融合
- ✅ 投影到 hidden_size (256)
- ✅ 使用 LayerNorm 归一化
- ❌ 不是 center/whiten

---

### Step 4: 与Item Embedding融合（`_get_fused_item_embeddings` 方法）

```python
# 4.1 获取item embedding
item_emb = self.item_embedding(item_ids)  # [B, 256]

# 4.2 获取投影后的文本特征
text_proj = self._project_text(text_raw)  # [B, 256]

# 4.3 融合（通过gate控制权重）
alpha = torch.sigmoid(self.text_gate_param)
fused = item_emb + alpha * text_proj

# 4.4 最终LayerNorm
if self.fused_item_norm is not None:
    fused = self.fused_item_norm(fused)    # LayerNorm([B, 256])

return fused
```

**当前状态：**
- ✅ 使用 LayerNorm 归一化
- ✅ 通过 gate 控制文本权重

---

## 📊 当前 vs 需求对比

| 处理阶段 | 需求 | 当前实现 | 符合度 |
|---------|------|---------|--------|
| **特征加载** | center/whiten + L2 | 仅 L2 归一化 | ⚠️ 部分 |
| **特征拼接** | 统一处理 | 拼接 base + llm | ✅ 是 |
| **投影到256** | 线性投影 | Cross + Deep + Linear | ✅ 是 |
| **投影后归一化** | LayerNorm | LayerNorm | ✅ 是 |
| **融合后归一化** | LayerNorm | LayerNorm | ✅ 是 |
| **交叉层** | 统一交叉层 | DCN Cross Network (2层) | ✅ 是 |

### 关键差异

#### ❌ 缺失：center/whiten

**需求：** 两路特征统一 center/whiten + L2/LayerNorm

**当前：** 
1. **加载阶段**：仅 L2 归一化（`normalize_text: true`）
2. **没有** center（中心化）
3. **没有** whiten（白化）

**影响：**
- 两路特征可能有不同的均值和方差分布
- 未去除特征间的相关性
- 可能影响对齐和融合效果

---

## ✅ 改进建议

### 方案A：在特征文件层面做 center/whiten（推荐）

**优点：**
- 离线处理，不增加训练开销
- 特征文件可复用
- 符合"统一标准化"的要求

**实施：**
```bash
# 1. 重新生成 TF-IDF 特征（已启用 whiten）
bash tools/gen_tfidf_only_fast.sh

# 2. 重新生成 Qwen3 特征（启用 whiten）
python tools/build_item_text_emb_qwen3_hf.py \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --dataset Amazon_Beauty \
  ...  # 确保启用 whiten
```

**配置调整：**
```yaml
# sasrec_align_qwen3.yaml
normalize_text: false  # 关闭模型内L2归一化（特征文件已whitened）
text_proj_norm: true   # 保留投影后LayerNorm
fused_item_norm: true  # 保留融合后LayerNorm
```

---

### 方案B：在模型内做 center/whiten

**优点：**
- 更灵活，可以动态调整
- 可以在训练集上计算统计量

**缺点：**
- 增加训练开销
- 需要修改模型代码

**实施：**

修改 `sasrec_align.py` 的 `__init__` 方法：

```python
# 在 L2 归一化之前添加 center + whiten
if self.normalize_text:
    with torch.no_grad():
        # 对每路特征分别做 center + whiten
        if emb_base is not None:
            emb_base = self._center_whiten_normalize(emb_base)
        if emb_llm is not None:
            emb_llm = self._center_whiten_normalize(emb_llm)

def _center_whiten_normalize(self, emb: torch.Tensor) -> torch.Tensor:
    """Center + Whiten + (optional) L2 normalize"""
    # 排除PAD（第0行）
    train_emb = emb[1:]
    
    # Step 1: Center
    mean = train_emb.mean(dim=0, keepdim=True)
    emb_centered = emb - mean
    emb_centered[0, :] = 0.0  # PAD保持零
    
    # Step 2: Whiten
    train_centered = train_emb - mean
    cov = (train_centered.T @ train_centered) / len(train_centered)
    
    # SVD: cov = U @ diag(S) @ U.T
    U, S, _ = torch.svd(cov)
    whiten_matrix = U @ torch.diag(1.0 / torch.sqrt(S + 1e-5))
    
    emb_whitened = emb_centered @ whiten_matrix
    emb_whitened[0, :] = 0.0
    
    # Step 3: (Optional) L2 normalize
    # 如果whitened，通常不需要L2归一化
    # norms = torch.norm(emb_whitened[1:], p=2, dim=1, keepdim=True)
    # emb_whitened[1:] = emb_whitened[1:] / norms.clamp_min(1e-8)
    
    return emb_whitened
```

---

## 🎯 推荐行动方案

### 短期（快速验证）

1. **使用方案A**：在特征文件层面做 center/whiten
   ```bash
   # 确保生成的特征已启用 whiten
   python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
   python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.npy
   ```

2. **调整模型配置**
   ```yaml
   normalize_text: false  # 特征已whitened，不需要再L2归一化
   text_proj_norm: true   # 保留LayerNorm
   ```

3. **重新训练**
   ```bash
   bash two_phase_run_tfidf_llm.sh
   ```

### 长期（完整实现）

1. **在模型内实现 center/whiten**（方案B）
2. **添加配置开关**
   ```yaml
   use_whiten: true              # 启用whiten
   whiten_before_concat: true    # 拼接前whiten
   use_layernorm_after_proj: true  # 投影后LayerNorm
   ```

3. **支持多种归一化组合**
   - center only
   - center + whiten
   - center + whiten + L2
   - LayerNorm only

---

## 📊 完整的理想流程

```
TF-IDF特征 [N, 256]         Qwen3特征 [N, 256]
    ↓                             ↓
Center + Whiten              Center + Whiten
    ↓                             ↓
(Optional) L2 Norm          (Optional) L2 Norm
    ↓                             ↓
    └──────── Concat ──────────┘
                ↓
           [B, 512]
                ↓
        Cross Network (DCN)
                ↓
         Deep Network
                ↓
         Concat + Linear
                ↓
           [B, 256]
                ↓
          LayerNorm  ← 投影后归一化
                ↓
    Fusion with Item Emb
                ↓
          LayerNorm  ← 融合后归一化
                ↓
           Output
```

---

## ⚠️ 当前流程的潜在问题

1. **两路特征未统一标准化**
   - TF-IDF 和 Qwen3 可能有不同的分布
   - 仅 L2 归一化不能解决方差和相关性问题

2. **拼接后维度爆炸**
   - 512维输入到 Cross Network，计算量大

3. **缺少中心化**
   - LayerNorm 会做中心化，但是在拼接之后
   - 理想情况是拼接前就统一

---

**分析日期**: 2025-12-03  
**结论**: 当前实现**部分符合**需求，缺少 center/whiten 步骤

