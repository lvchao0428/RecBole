# MV-Align Cover Image Material for Nano Banana

## 三段式 Cover Image 设计

---

## 左侧：现实问题 (Cold-Start Item)

### 示例：新上架美妆商品

```
📦 New Item Example
━━━━━━━━━━━━━━━━━━━━━━━━━━
Title: "Vitamin C Brightening Serum"
Description: "30ml facial serum..."

Interactions: ~0 (cold-start)
Stratum: new [1, 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: No collaborative signal,
         ID embedding under-trained
```

---

## 中间：MV-Align 方案 (四个关键模块)

### 1️⃣ Prepare: Normalize + Whitening

**功能**: Multi-view decorrelate，解决 view 间冗余

```python
# Per-view ZCA Whitening (decorrelate each view)
# Code from sasrecalignmultiviewv2.py

def _gather_text_views(self, ids_flat):
    for idx, proj in enumerate(self.text_view_proj):
        # 1. Load raw view embeddings
        gathered = view_emb_table[ids_flat]  # [B, view_dim]
        
        # 2. Project to hidden_size
        projected = proj(gathered)  # [B, 256]
        
        # 3. SENet enhancement (channel attention)
        if self.use_text_view_senet:
            excitation = self.text_view_senet[idx](projected)
            refined = projected * excitation
        
        # 4. Per-view L2 normalization
        if self.per_view_l2_norm:
            refined = F.normalize(refined, p=2, dim=-1)
```

**关键公式** (ZCA Whitening):
```
X_centered = X - μ
Σ = (1/n) X_centered^T X_centered  
W = (Σ + εI)^(-1/2)
X_whitened = X_centered × W
```

---

### 2️⃣ Multi-view: Identity / Function / Audience / Category

**功能**: 4个 LLM prompt 视角，捕获多维语义

```python
# 4 Prompt Views (from build_item_text_emb_qwen3_hf.py)
PROMPT_PRESETS = {
    "multiview": [
        # View 0: Identity/Title (Base)
        "Identify the item: [TITLE] {text}",
        
        # View 1: Function/Utility
        "What are the main functions and features of [TITLE] {text}?",
        
        # View 2: Target Audience
        "Who is the target audience or user group for [TITLE] {text}?",
        
        # View 3: Category/Context
        "Categorize the item [TITLE] {text} and describe its context.",
    ],
}
```

**Embedding Budget**:
```
Single-View: 1 × 256-d
Multi-View:  4 × 64-d (same total budget!)
+ TF-IDF 256-d shared
```

---

### 3️⃣ Cross: Explicit ID × Text Interaction

**功能**: DCN-V2 style cross network，捕获高阶交叉特征

```python
# DCN-V2 Cross Network (from sasrec_align.py)
class DCNV2Cross(nn.Module):
    """x_{l+1} = x_l + x_0 ⊙ (W_l x_l + b_l)"""
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x0_u = x0.unsqueeze(dim=2)  # [B, D, 1]
        xl = x0_u
        for i in range(self.num_layers):
            xl_w = torch.matmul(self.cross_layer_w[i], xl)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x0_u, xl_w)  # Element-wise cross
            xl = xl_dot + xl  # Residual
        return xl.squeeze(dim=2)

# Fusion: ID Embedding + Text Features → Cross Network
fusion_input = torch.cat([id_emb, text_proj], dim=1)  # [B, 512]
cross_out = self.item_fusion_cross(fusion_input)
```

---

### 4️⃣ Align: Text→ID Contrastive + Cold-Start Reweighting

**功能**: InfoNCE 对齐 + 防止高频商品主导

```python
# Multi-View Alignment Loss with Cold-Start Reweighting
# (from sasrecalignmultiviewv2.py)

def _info_nce_align_weighted(self, a, b, sample_weights):
    """Weighted InfoNCE: prevent head items dominating"""
    a = F.normalize(a, dim=1)  # ID embeddings
    b = F.normalize(b, dim=1)  # Text embeddings
    
    sim = torch.matmul(a, b.t()) / self.temperature  # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    
    per_sample_loss = F.cross_entropy(sim, labels, reduction='none')
    
    # Cold-start reweighting: low-freq items get higher weight
    weighted_loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return weighted_loss

def _compute_cold_start_weights(self, item_ids):
    """Weight formula: w_i = 1 + boost × max(0, threshold - pop_i) / threshold"""
    item_pop = self.item_popularity[item_ids].float()
    cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold
    weights = 1.0 + self.cold_start_align_boost * cold_factor
    return weights
```

**对齐损失公式**:
```
L_align = Σ_v w_v · InfoNCE(e^ID, v)

InfoNCE(a, b) = -log(exp(sim(a,b)/τ) / Σ_j exp(sim(a,b_j)/τ))

Cold-start weight: w_i = 1 + β × max(0, P₀ - pop_i) / P₀
```

---

## 右侧：效果 (Full Ranking + Cold-Start Gains)

### 主要指标 (Amazon Toys, 7B Aggressive)

| Metric | Baseline (TF-IDF) | MV-Align | Δ |
|--------|-------------------|----------|---|
| **HR@10** | 6.14% | **6.92%** | +12.7% |
| **MRR@10** | 3.31% | **3.68%** | +11.2% |
| **HR_new@10** | 2.04% | **4.08%** | +100%↑ |

### 关键结论
```
✅ Full-ranking gains (not sampled eval!)
✅ 2× improvement on cold-start items
✅ No sacrifice on frequent items
✅ Deployable plug-in recipe
```

---

## 视觉设计建议

### 配色方案
- **左侧 (Problem)**: 灰色/红色调 (问题感)
- **中间 (Solution)**: 
  - Prepare: 蓝色
  - Multi-view: 渐变色 (粉/蓝/绿/橙 for 4 views)
  - Cross: 紫色
  - Align: 橙色
- **右侧 (Results)**: 绿色 (成功感)

### 布局
```
┌─────────────┬─────────────────────────────────┬─────────────┐
│   Problem   │          MV-Align Solution       │   Results   │
│             │                                  │             │
│  Cold-start │  ① Prepare  ② Multi-view        │  Full-rank  │
│  item with  │  ③ Cross    ④ Align             │  HR +12.7%  │
│  ~0 clicks  │                                  │             │
│             │  Budget: 1×256 = 4×64            │  Cold-start │
│  new:[1,3)  │  (same total embedding size!)   │  HR +100%↑  │
└─────────────┴─────────────────────────────────┴─────────────┘
```

---

## 核心卖点 (One-liner)

> **MV-Align: A deployable text plug-in recipe for sequential recommendation, delivering reliable full-ranking and cold-start gains without extra embedding budget.**
