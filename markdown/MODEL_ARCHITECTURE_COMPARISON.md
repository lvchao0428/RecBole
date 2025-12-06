# SASRecAlign 模型架构对比分析

对比 `two_phase_run_tfidf.sh` vs `two_phase_run_tfidf_llm.sh` 的模型结构

---

## 📋 配置对比

| 特性 | TF-IDF Only | TF-IDF + LLM |
|------|-------------|--------------|
| **脚本** | `two_phase_run_tfidf.sh` | `two_phase_run_tfidf_llm.sh` |
| **配置文件** | `sasrec_align_base.yaml` | `sasrec_align_qwen3.yaml` |
| **use_llm** | `false` | `true` |
| **文本特征路径** | `item_text_emb.base.npy` | `item_text_emb.base.npy` + `item_text_emb.qwen3.npy` |
| **文本模式** | `base` | `both` |
| **原始文本维度** | 256 | 512 (256 + 256) |

---

## 🔄 完整架构对比图

### 架构 A: TF-IDF Only (`use_llm: false`)

```
┌─────────────────────────────────────────────────────────────────┐
│                      输入：用户序列                                │
│                  item_seq: [B, L]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   1. ID Embedding 路径                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   item_emb                                 position_emb
   [B, L, 256]                               [B, L, 256]
        │                                           │
        └──────────────────┬────────────────────────┘
                           ↓
                    input_emb = item_emb + position_emb
                         [B, L, 256]
                           ↓
                      LayerNorm
                           ↓
                    Dropout (0.5)
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│            2. Transformer Encoder（序列建模）                      │
│         2 layers × 2 heads × 256 hidden                         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
                    trm_output[-1]
                      [B, L, 256]
                           ↓
                gather_indexes(seq_len - 1)
                           ↓
                    seq_output
                      [B, 256]  ← 序列表示（ID特征）


┌─────────────────────────────────────────────────────────────────┐
│                   3. 文本特征路径（单路TF-IDF）                     │
└─────────────────────────────────────────────────────────────────┘

    item_text_emb_base
         [N, 256]
            ↓
      L2 Normalize
    (normalize_text=true)
            ↓
     text_raw [B, 256]  ← _text_mode = "base"

┌─────────────────────────────────────────────────────────────────┐
│          4. 文本投影网络（Cross + Deep）                            │
│             text_in_dim = 256                                   │
└─────────────────────────────────────────────────────────────────┘
            ↓
    ┌───────┴───────┐
    │               │
Cross Network   Deep Network
 (DCN-V2)         (MLP)
  [B, 256]       [B, 256]
    │               │
    │          3 layers MLP
    │          [256, 256, 256]
    │          dropout=0.0
    │               │
Dropout (0.5)       │
    │               │
    └───────┬───────┘
            ↓
    Concat [B, 512]
            ↓
  Linear Predictor
    (512 → 256)
            ↓
     LayerNorm
   (text_proj_norm)
            ↓
   text_proj [B, 256]  ← 文本表示


┌─────────────────────────────────────────────────────────────────┐
│              5. Item Embedding 融合（候选Item）                    │
│          fusion_input_dim = 256 + 256 = 512                     │
└─────────────────────────────────────────────────────────────────┘

      对于候选 item_ids:
                   ↓
            item_embedding
              [B, 256]
                   ↓
        ┌──────────┴──────────┐
        │                     │
    item_emb          text_features (base)
    [B, 256]              [B, 256]
        │                     │
        │               text_gate ×
        │              effective_weight
        │                     ↓
        │              scaled_text [B, 256]
        │                     │
        └──────────┬───────────┘
                   ↓
           Concat [B, 512]  ← fusion_input_dim
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2)                   (MLP)
Input: [B, 512]            [512, 256, 256]
Output: [B, 512]          Output: [B, 256]
      │                         │
  Dropout (0.5)                 │
      │                         │
      └──────────┬──────────────┘
                 ↓
          Concat [B, 768]
                 ↓
       Linear Predictor
         (768 → 256)
                 ↓
          LayerNorm
        (fused_item_norm)
                 ↓
       fused_item_emb
          [B, 256]  ← 融合后的Item表示


┌─────────────────────────────────────────────────────────────────┐
│                    6. 打分与预测                                  │
└─────────────────────────────────────────────────────────────────┘

    seq_output              fused_item_emb
     [B, 256]                  [B, 256]
        │                          │
        ├─> F.normalize ←──────────┤  (cosine_score=true)
        │                          │
        └──────────┬───────────────┘
                   ↓
        score = 10.0 × (seq_output · fused_item_emb)
                   ↓
              [B] scores
```

---

### 架构 B: TF-IDF + LLM (`use_llm: true`)

```
┌─────────────────────────────────────────────────────────────────┐
│                      输入：用户序列                                │
│                  item_seq: [B, L]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   1. ID Embedding 路径                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   item_emb                                 position_emb
   [B, L, 256]                               [B, L, 256]
        │                                           │
        └──────────────────┬────────────────────────┘
                           ↓
                    input_emb = item_emb + position_emb
                         [B, L, 256]
                           ↓
                      LayerNorm
                           ↓
                    Dropout (0.5)
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│            2. Transformer Encoder（序列建模）                      │
│         2 layers × 2 heads × 256 hidden                         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
                    trm_output[-1]
                      [B, L, 256]
                           ↓
                gather_indexes(seq_len - 1)
                           ↓
                    seq_output
                      [B, 256]  ← 序列表示（ID特征）


┌─────────────────────────────────────────────────────────────────┐
│                   3. 文本特征路径（双路）                           │
└─────────────────────────────────────────────────────────────────┘

    TF-IDF Path                      Qwen3 Path
        ↓                                 ↓
item_text_emb_base               item_text_emb_llm
  [N, 256]                           [N, 256]
        ↓                                 ↓
  L2 Normalize                      L2 Normalize
 (normalize_text)                 (normalize_text)
        ↓                                 ↓
        └─────────────┬───────────────────┘
                      ↓
              Concat [B, 512]  ← _text_mode = "both"

┌─────────────────────────────────────────────────────────────────┐
│          4. 文本投影网络（Cross + Deep）                            │
│             text_in_dim = 512                                   │
└─────────────────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        │                           │
   Cross Network              Deep Network
    (DCN-V2)                    (MLP)
   [B, 512]                 [512, 256, 256]
        │                    Output: [B, 256]
   Dropout (0.5)                    │
        │                           │
        └──────────┬─────────────────┘
                   ↓
            Concat [B, 768]
                   ↓
         Linear Predictor
           (768 → 256)
                   ↓
            LayerNorm
           (text_proj_norm)
                   ↓
           text_proj [B, 256]  ← 文本表示


┌─────────────────────────────────────────────────────────────────┐
│              5. Item Embedding 融合（候选Item）                    │
│          fusion_input_dim = 256 + 512 = 768                     │
└─────────────────────────────────────────────────────────────────┘

      对于候选 item_ids:
                   ↓
            item_embedding
              [B, 256]
                   ↓
        ┌──────────┴──────────┐
        │                     │
    item_emb          text_features (base+llm)
    [B, 256]               [B, 512]
        │                     │
        │               text_gate ×
        │              effective_weight
        │                     ↓
        │              scaled_text [B, 512]
        │                     │
        └──────────┬───────────┘
                   ↓
           Concat [B, 768]  ← fusion_input_dim
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2)                   (MLP)
Input: [B, 768]            [768, 256, 256]
Output: [B, 768]          Output: [B, 256]
      │                         │
  Dropout (0.5)                 │
      │                         │
      └──────────┬──────────────┘
                 ↓
          Concat [B, 1024]
                 ↓
       Linear Predictor
        (1024 → 256)
                 ↓
          LayerNorm
        (fused_item_norm)
                 ↓
       fused_item_emb
          [B, 256]  ← 融合后的Item表示


┌─────────────────────────────────────────────────────────────────┐
│                    6. 打分与预测                                  │
└─────────────────────────────────────────────────────────────────┘

    seq_output              fused_item_emb
     [B, 256]                  [B, 256]
        │                          │
        ├─> F.normalize ←──────────┤  (cosine_score=true)
        │                          │
        └──────────┬───────────────┘
                   ↓
        score = 10.0 × (seq_output · fused_item_emb)
                   ↓
              [B] scores
```

---

## 📊 关键差异总结

### 1. 文本特征维度

| 组件 | TF-IDF Only | TF-IDF + LLM |
|------|-------------|--------------|
| **文本输入维度** (`text_in_dim`) | 256 | **512** |
| **文本模式** (`_text_mode`) | `"base"` | `"both"` |
| **文本特征拼接** | base[256] | base[256] + llm[256] |

### 2. 文本投影网络带宽差异

#### Text Projection (文本投影)

| 模块 | TF-IDF Only | TF-IDF + LLM | 带宽比 |
|------|-------------|--------------|--------|
| **Cross Network Input** | 256 | **512** | 2x |
| **Cross Network Output** | 256 | **512** | 2x |
| **Deep Network Input** | 256 | **512** | 2x |
| **Deep Network Layers** | [256, 256, 256] | [**512**, 256, 256] | 2x at input |
| **Deep Network Output** | 256 | 256 | 1x |
| **Concat Before Predictor** | 512 (256+256) | **768** (512+256) | 1.5x |
| **Predictor Linear** | 512 → 256 | **768** → 256 | 1.5x |

### 3. Item融合网络带宽差异

#### Item Fusion (Item侧融合)

| 模块 | TF-IDF Only | TF-IDF + LLM | 带宽比 |
|------|-------------|--------------|--------|
| **Item Embedding** | 256 | 256 | 1x |
| **Text Features** | 256 | **512** | 2x |
| **Fusion Input Dim** | 512 (256+256) | **768** (256+512) | 1.5x |
| **Fusion Cross Input** | 512 | **768** | 1.5x |
| **Fusion Cross Output** | 512 | **768** | 1.5x |
| **Fusion Deep Input** | 512 | **768** | 1.5x |
| **Fusion Deep Layers** | [512, 256, 256] | [**768**, 256, 256] | 1.5x at input |
| **Fusion Deep Output** | 256 | 256 | 1x |
| **Concat Before Predictor** | 768 (512+256) | **1024** (768+256) | 1.33x |
| **Fusion Predictor** | 768 → 256 | **1024** → 256 | 1.33x |

---

## 🔢 参数量对比

### ID Feature 参数（两个架构相同）

| 组件 | 参数量 | 备注 |
|------|--------|------|
| `item_embedding` | N × 256 | N = 物品数量（Amazon_Beauty ≈ 5,000） |
| `position_embedding` | 50 × 256 = 12,800 | MAX_ITEM_LIST_LENGTH = 50 |
| `Transformer Encoder` | ~0.5M | 2 layers × 2 heads × 256 hidden |
| `LayerNorm` (input) | 512 | 256×2 (weight + bias) |
| **Total ID Params** | **~1.3M + N×256** | ≈ **2.6M** (N=5000时) |

### Text Feature 参数对比

#### Text Projection Networks

| 组件 | TF-IDF Only | TF-IDF + LLM |
|------|-------------|--------------|
| **Cross Network** | | |
| - DCNV2Cross(256, 2 layers) | ~0.5K | - |
| - DCNV2Cross(512, 2 layers) | - | ~1K |
| **Deep Network** | | |
| - Layer 1: 256→256 | 65,792 | - |
| - Layer 1: 512→256 | - | **131,328** |
| - Layer 2: 256→256 | 65,792 | 65,792 |
| - Layer 3: 256→256 | 65,792 | 65,792 |
| **Text Predictor** | | |
| - Linear: 512→256 | 131,328 | - |
| - Linear: 768→256 | - | **196,864** |
| **Text Proj LayerNorm** | 512 | 512 |
| **Subtotal** | ~329K | **~460K** |

#### Item Fusion Networks

| 组件 | TF-IDF Only | TF-IDF + LLM |
|------|-------------|--------------|
| **Fusion Cross Network** | | |
| - DCNV2Cross(512, 2 layers) | ~1K | - |
| - DCNV2Cross(768, 2 layers) | - | ~1.5K |
| **Fusion Deep Network** | | |
| - Layer 1: 512→256 | 131,328 | - |
| - Layer 1: 768→256 | - | **196,864** |
| - Layer 2: 256→256 | 65,792 | 65,792 |
| - Layer 3: 256→256 | 65,792 | 65,792 |
| **Fusion Predictor** | | |
| - Linear: 768→256 | 196,864 | - |
| - Linear: 1024→256 | - | **262,400** |
| **Item Emb LayerNorm** | 512 | 512 |
| **Fused Item LayerNorm** | 512 | 512 |
| **Subtotal** | ~460K | **~591K** |

#### Text Gate Parameters

| 组件 | TF-IDF Only | TF-IDF + LLM |
|------|-------------|--------------|
| `text_gate_param` | 1 | 1 |
| **Subtotal** | 1 | 1 |

#### Total Text Feature Parameters

| 配置 | 参数量 | 增加 |
|------|--------|------|
| **TF-IDF Only** | ~789K | - |
| **TF-IDF + LLM** | **~1,051K** | **+33%** |

### 总参数量对比

| 配置 | ID Params | Text Params | Total | 比例 |
|------|-----------|-------------|-------|------|
| **TF-IDF Only** | 2.6M | 0.79M | **3.39M** | 100% |
| **TF-IDF + LLM** | 2.6M | 1.05M | **3.65M** | **107.7%** |

**增量**: +262K 参数 (+7.7%)

---

## ⚖️ 模型带宽一致性分析

### 结论：**带宽不一致** ❌

两个模型的带宽**不一致**，主要体现在：

1. **文本通路带宽**：
   - TF-IDF Only: 256维文本输入
   - TF-IDF + LLM: 512维文本输入（**2倍带宽**）

2. **中间层带宽**：
   - 文本投影网络的 Cross Network: 256 vs 512 (**2倍**)
   - Item融合网络输入: 512 vs 768 (**1.5倍**)
   - 融合后拼接: 768 vs 1024 (**1.33倍**)

3. **瓶颈统一**：
   - 两个模型在最终输出时统一到 256维
   - 序列表示(seq_output)始终是 256维
   - 融合Item表示(fused_item_emb)始终是 256维

### 带宽放大倍数总结

| 路径阶段 | TF-IDF Only | TF-IDF + LLM | 倍数 |
|---------|-------------|--------------|------|
| **文本原始输入** | 256 | 512 | **2.0x** |
| **文本投影Cross** | 256 | 512 | **2.0x** |
| **文本投影拼接** | 512 | 768 | **1.5x** |
| **Item融合输入** | 512 | 768 | **1.5x** |
| **Item融合Cross** | 512 | 768 | **1.5x** |
| **Item融合拼接** | 768 | 1024 | **1.33x** |
| **最终Item表示** | 256 | 256 | **1.0x** |

---

## 🎯 ID Feature vs Text Feature 参数量对比

### TF-IDF Only 配置

```
ID Feature:    2.6M  (76.7%)
Text Feature:  0.79M (23.3%)
─────────────────────────
Total:         3.39M (100%)
```

- **ID占主导**：76.7% vs 23.3%
- **比例**: 3.3:1

### TF-IDF + LLM 配置

```
ID Feature:    2.6M  (71.2%)
Text Feature:  1.05M (28.8%)
─────────────────────────
Total:         3.65M (100%)
```

- **ID仍占主导**：71.2% vs 28.8%
- **比例**: 2.5:1

### 对比结论

1. **ID特征参数**在两个配置中完全相同（2.6M）
2. **Text特征参数**增加了33%（0.79M → 1.05M）
3. **总体参数**增加了7.7%（3.39M → 3.65M）
4. ID与Text的**参数比例**从 3.3:1 降到 2.5:1
5. 即使加入LLM特征，**ID特征仍占主导**（71.2%）

---

## 💡 设计洞察

### 1. 渐进式信息压缩

两个架构都采用了"漏斗式"设计：

- **TF-IDF Only**: 256 → 512 → 768 → **256**
- **TF-IDF + LLM**: 512 → 768 → 1024 → **256**

最终都压缩到256维统一表示空间。

### 2. Cross Network的输入维度自适应

DCN-V2 Cross Network的输入维度会根据文本特征维度自动调整：
- 适应不同的文本输入带宽
- 但会增加参数量

### 3. 参数增长的主要来源

参数增量主要来自：
1. **Fusion Deep Network 第一层**：512→256 vs 768→256 (+65K)
2. **Fusion Predictor**：768→256 vs 1024→256 (+65K)
3. **Text Deep Network 第一层**：256→256 vs 512→256 (+65K)
4. **Text Predictor**：512→256 vs 768→256 (+65K)

总增量 ≈ 262K

### 4. 模型容量 vs 特征信息

- LLM特征带来额外信息（语义理解）
- 需要更大的网络容量来处理（+33%参数）
- 但最终表示空间保持一致（256维）

---

## 📝 建议

### 1. 为了公平对比

如果想让两个模型有相同的参数量/带宽，可以：

**选项A：扩展 TF-IDF Only 模型**
```yaml
# 人工加倍 base 特征维度
# 或添加随机/padding特征使其达到512维
```

**选项B：压缩 TF-IDF + LLM 模型**
```yaml
# 先将 base 和 llm 各自投影到 128维
# 然后拼接成 256维
```

### 2. 当前配置的合理性

当前配置是合理的：
- LLM特征包含更丰富信息，需要更大容量
- 参数增量仅7.7%，开销可接受
- 最终表示空间统一，便于对比效果

### 3. 性能提升的归因

如果 TF-IDF + LLM 性能更好，来源可能是：
1. **LLM语义特征** (主要)
2. **更大的网络容量** (次要，仅+7.7%)
3. 两者的协同效应

---

## 🔀 架构 C: MultiView (4-view split + base)

### 配置概览

| 特性 | 值 |
|------|-----|
| **脚本** | `two_phase_run_multiview_split.sh` |
| **配置文件** | `sasrec_align_multi_view.yaml` |
| **模型类** | `SASRecAlignMultiView` |
| **文本特征** | base (TF-IDF) [256] + 4-view Qwen3 [4×64] |
| **总文本维度** | 256 + 256 = **512** |

### 完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      输入：用户序列                                │
│                  item_seq: [B, L]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   1. ID Embedding 路径                            │
│                   (与 A/B 完全相同)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    seq_output [B, 256]  ← 序列表示


┌─────────────────────────────────────────────────────────────────┐
│              3. 文本特征路径（Multi-View + Base）                   │
└─────────────────────────────────────────────────────────────────┘

    TF-IDF Base              4-View Qwen3 Embeddings
        ↓                         ↓
item_text_emb_base      ┌─────────┴─────────┐
  [N, 256]              │                   │
        │           View 0   View 1   View 2   View 3
        │         (identity)(function)(audience)(category)
        │           [N,64]   [N,64]   [N,64]   [N,64]
        │               ↓       ↓       ↓       ↓
        │           ┌───┴───┬───┴───┬───┴───┬───┴───┐
        │           │       │       │       │       │
        │        Linear  Linear  Linear  Linear
        │         64→256  64→256  64→256  64→256
        │           │       │       │       │       │
        │           ↓       ↓       ↓       ↓
        │        SENet   SENet   SENet   SENet
        │        [256]   [256]   [256]   [256]
        │           │       │       │       │
        │           ↓       ↓       ↓       ↓
        │        Gate_0  Gate_1  Gate_2  Gate_3
        │         (w0)    (w1)    (w2)    (w3)
        │           │       │       │       │
        │           └───┬───┴───┬───┴───┬───┘
        │               ↓       ↓       ↓
        │           weighted & concat
        │             [B, 1024]
        │               │
        └───────────────┴───────────────┐
                        ↓
                Concat [B, 1280]
            (base[256] + views[1024])
                        ↓

┌─────────────────────────────────────────────────────────────────┐
│          4. Multi-View Concat Projection                        │
└─────────────────────────────────────────────────────────────────┘
                        ↓
            Linear: 1280 → 512
                        ↓
            multiview_concat_proj
                  [B, 512]  ← 文本表示
                  
        注意：跳过独立的Text Projection (Cross+Deep)
             直接使用 concat_proj 的输出


┌─────────────────────────────────────────────────────────────────┐
│              5. Item Embedding 融合（候选Item）                    │
│          fusion_input_dim = 256 + 512 = 768                     │
│               (与架构 B 完全相同)                                  │
└─────────────────────────────────────────────────────────────────┘

      对于候选 item_ids:
                   ↓
            item_embedding
              [B, 256]
                   ↓
        ┌──────────┴──────────┐
        │                     │
    item_emb        multiview_concat_proj
    [B, 256]               [B, 512]
        │                     │
        │               text_gate ×
        │              effective_weight
        │                     ↓
        │              scaled_text [B, 512]
        │                     │
        └──────────┬───────────┘
                   ↓
           Concat [B, 768]  ← fusion_input_dim
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2)                   (MLP)
Input: [B, 768]            [768, 256, 256]
Output: [B, 768]          Output: [B, 256]
      │                         │
  Dropout (0.5)                 │
      │                         │
      └──────────┬──────────────┘
                 ↓
          Concat [B, 1024]
                 ↓
       Linear Predictor
        (1024 → 256)
                 ↓
          LayerNorm
        (fused_item_norm)
                 ↓
       fused_item_emb
          [B, 256]  ← 融合后的Item表示


┌─────────────────────────────────────────────────────────────────┐
│                    6. 打分与预测                                  │
│                   (与 A/B 完全相同)                               │
└─────────────────────────────────────────────────────────────────┘

    seq_output              fused_item_emb
     [B, 256]                  [B, 256]
        │                          │
        ├─> F.normalize ←──────────┤  (cosine_score=true)
        │                          │
        └──────────┬───────────────┘
                   ↓
        score = 10.0 × (seq_output · fused_item_emb)
                   ↓
              [B] scores
```

### 三架构详细对比表

| 组件 | TF-IDF Only | TF-IDF + LLM | MultiView | 说明 |
|------|-------------|--------------|-----------|------|
| **文本原始特征** | | | | |
| - Base (TF-IDF) | 256 | 256 | 256 | ✅ 全部相同 |
| - LLM features | - | 256 (整体) | 4×64=256 (分离) | ⚠️ 总量相同，形式不同 |
| - 总文本维度 | 256 | 512 | 512 | ✅ B/C 相同 |
| **文本预处理** | | | | |
| - Per-view Linear | - | - | 4×(64→256) | ❌ C 独有 |
| - Per-view SENet | - | - | 4×SENet(256) | ❌ C 独有 |
| - Per-view Gate | - | - | 4 params | ❌ C 独有 |
| - Concat dimension | 256 | 512 | 1024 + 256 = 1280 | ❌ 差异大 |
| - Concat Projection | - | - | 1280→512 | ❌ C 独有 |
| **Text Projection** | | | | |
| - Cross Network | 256 | 512 | - (跳过) | ⚠️ C 无此阶段 |
| - Deep Network | [256,256,256] | [512,256,256] | - (跳过) | ⚠️ C 无此阶段 |
| - Predictor | 512→256 | 768→256 | - (跳过) | ⚠️ C 无此阶段 |
| - Text Output | 256 | 256 | 512 | ⚠️ C 直接512 |
| **Item Fusion** | | | | |
| - Fusion Input | 512 | 768 | 768 | ✅ B/C 相同 |
| - Fusion Cross | 512 | 768 | 768 | ✅ B/C 相同 |
| - Fusion Deep | [512,256,256] | [768,256,256] | [768,256,256] | ✅ B/C 相同 |
| - Fusion Concat | 768 | 1024 | 1024 | ✅ B/C 相同 |
| - Fusion Output | 256 | 256 | 256 | ✅ 全部相同 |
| **Per-View Alignment** | - | - | ✅ 4个独立损失 | ❌ C 独有 |

### MultiView 关键特性

1. **细粒度视角控制**
   - 4个独立的prompt视角（identity, function, audience, category）
   - 每个视角独立处理和增强

2. **SENet动态增强**
   - 每个view经过SENet进行特征重标定
   - 动态调整特征权重

3. **可学习gate融合**
   - 4个可学习的gate参数控制view重要性
   - 自动学习最优view组合

4. **Per-view对齐损失**
   - 4个独立的InfoNCE对齐损失
   - 每个view与ID embedding分别对齐
   - 可学习的对齐权重

5. **架构对齐设计**
   - Concat projection输出512维，与TF-IDF+LLM对齐
   - Item Fusion阶段完全一致（768→256）

### 参数量详细对比

| 组件 | TF-IDF Only | TF-IDF + LLM | MultiView |
|------|-------------|--------------|-----------|
| **ID Feature** | 2.6M | 2.6M | 2.6M |
| **Text Feature** | | | |
| - Per-view Proj | - | - | 4×16,640 = 66,560 |
| - Per-view SENet | - | - | 4×32,768 = 131,072 |
| - Per-view Gates | - | - | 4 |
| - Per-view Align | - | - | 4 |
| - Concat Proj | - | - | 655,360 |
| - Text Cross | ~0.5K | ~1K | - |
| - Text Deep | ~197K | ~262K | - |
| - Text Predictor | 131K | 197K | - |
| - Fusion Networks | ~460K | ~591K | ~591K |
| **Total Text** | 0.79M | 1.05M | **1.44M** |
| **Grand Total** | 3.39M | 3.65M | **4.04M** |
| **vs TF-IDF+LLM** | -7.1% | baseline | **+10.7%** |

### 计算复杂度对比 (Per Item)

| 操作 | TF-IDF Only | TF-IDF + LLM | MultiView |
|------|-------------|--------------|-----------|
| **Text Processing** | | | |
| - Per-view Linear | - | - | 4×(64×256) = 65K |
| - Per-view SENet | - | - | 4×(256×64+64×256) = 131K |
| - Concat Proj | - | - | 1280×512 = 655K |
| - Text Cross | ~65K | ~262K | - |
| - Text Deep | ~197K | ~262K | - |
| - Text Predictor | ~131K | ~197K | - |
| **Text Subtotal** | ~393K | ~721K | **~851K** |
| **Item Fusion** | ~460K | ~591K | ~591K |
| **Total** | ~853K | ~1312K | **~1442K** |
| **vs TF-IDF+LLM** | -35% | baseline | **+10%** |

---

## 🎯 三架构终极对比

### 网络带宽演化

```
A. TF-IDF Only:
   256 (base) → 512 (concat) → 768 (fusion) → 256 (output)

B. TF-IDF + LLM:
   512 (base+llm) → 768 (text proj) → 768 (fusion) → 256 (output)
   ↑2x                ↑1.5x

C. MultiView:
   4×64 (views) → 4×256 (proj) → 1024 (concat) → 1280 (with base)
   → 512 (concat proj) → 768 (fusion) → 256 (output)
   ↑4个并行          ↑SENet增强      ↑2.5x           ↑与B对齐
```

### 信息流对比

| 阶段 | TF-IDF Only | TF-IDF + LLM | MultiView |
|------|-------------|--------------|-----------|
| **原始文本** | 256 | 512 | 512 (总量) |
| **特征形式** | 单一base | base + 整体LLM | base + 4-view LLM |
| **预处理** | 无 | 无 | SENet增强 |
| **文本融合** | 直接使用 | concat | 加权concat |
| **文本投影** | Cross+Deep | Cross+Deep | Linear (跳过Cross+Deep) |
| **到融合维度** | 512 | 768 | 768 |
| **融合方式** | Cross+Deep | Cross+Deep | Cross+Deep (相同) |

### 设计哲学对比

| 维度 | TF-IDF Only | TF-IDF + LLM | MultiView |
|------|-------------|--------------|-----------|
| **文本特征** | 统计特征 | 整体语义 | 多视角语义 |
| **处理方式** | 简单 | 中等 | 复杂 |
| **特征增强** | 无 | 无 | SENet |
| **融合策略** | 单一 | 双路 | 多视角加权 |
| **对齐策略** | 无/单一 | 整体 | Per-view |
| **参数效率** | 高 | 中 | 低 |
| **表达能力** | 低 | 高 | 最高 |

### 适用场景

**TF-IDF Only**:
- ✅ 快速原型验证
- ✅ 计算资源受限
- ✅ 简单推荐场景
- ❌ 复杂语义理解

**TF-IDF + LLM**:
- ✅ 平衡性能和效率
- ✅ 整体语义理解
- ✅ 生产环境部署
- ⚠️ 单一LLM视角

**MultiView**:
- ✅ 最大化性能
- ✅ 多视角语义理解
- ✅ 研究和实验
- ❌ 计算开销大
- ❌ 参数量大

---

**文档日期**: 2025-12-04  
**对比模型**: SASRecAlign (TF-IDF Only vs TF-IDF+LLM vs MultiView)  
**配置文件**: `sasrec_align_base.yaml` vs `sasrec_align_qwen3.yaml` vs `sasrec_align_multi_view.yaml`

