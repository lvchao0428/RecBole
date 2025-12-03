# SASRecAlign 完整模型架构详解

基于 `two_phase_run_tfidf_llm.sh` 的双路文本特征配置

---

## 📊 完整数据流图

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
                       Dropout
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│            2. Transformer Encoder（序列建模）                      │
│         SASRec Transformer (self-attention layers)              │
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
 (if normalize_text)              (if normalize_text)
        ↓                                 ↓
        └─────────────┬───────────────────┘
                      ↓
              Concat [B, 512]
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│            4. 文本投影网络（Cross + Deep）                          │
└─────────────────────────────────────────────────────────────────┘
                      ↓
             (Optional) SENet
              Amplifier [B, 512]
                      ↓
        ┌─────────────┴─────────────┐
        │                           │
   Cross Network              Deep Network
    (DCN-V2)                    (MLP)
   [B, 512]                    [B, 256]
        │                           │
   Dropout                          │
        │                           │
        └──────────┬─────────────────┘
                   ↓
            Concat [B, 768]
                   ↓
         Linear Predictor
              [B, 256]
                   ↓
            LayerNorm
           (text_proj_norm)
                   ↓
           text_proj [B, 256]  ← 文本表示


┌─────────────────────────────────────────────────────────────────┐
│              5. Item Embedding 融合（候选Item）                    │
└─────────────────────────────────────────────────────────────────┘

      对于候选item_ids:
                   ↓
            item_embedding
              [B, 256]
                   ↓
          (Optional) Normalize
                   ↓
        ┌──────────┴──────────┐
        │                     │
    item_emb            text_features
    [B, 256]               [B, 512]
        │                  (base + llm)
        │                     ↓
        │              (Optional) SENet
        │                     ↓
        │               text_gate ×
        │              effective_weight
        │                     ↓
        │              scaled_text [B, 512]
        │                     │
        └──────────┬───────────┘
                   ↓
           Concat [B, 768]
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2)                   (MLP)
   [B, 768]                 [B, 256]
      │                         │
  Dropout                       │
      │                         │
      └──────────┬──────────────┘
                 ↓
          Concat [B, 1024]
                 ↓
       Linear Predictor
            [B, 256]
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
        │  (if cosine_score)       │
        ├─> F.normalize ←──────────┤
        │                          │
        └──────────┬───────────────┘
                   ↓
            score = cosine_scale × (seq_output · fused_item_emb)
                         or
                score = seq_output · fused_item_emb
                   ↓
              [B] scores
```

---

## 🔧 详细模块说明

### 1. ID Embedding 路径（序列建模）

```python
# 代码位置：forward() 方法，864-898行

# Step 1.1: Token Dropout（训练时序列增强）
if self.training and self.token_dropout_prob > 0.0:
    item_seq = apply_token_dropout(item_seq)  # 随机mask一些token

# Step 1.2: Position Embedding
position_ids = torch.arange(L).expand(B, L)        # [B, L]
position_embedding = self.position_embedding(position_ids)  # [B, L, 256]

# Step 1.3: Item Embedding
item_emb = self.item_embedding(item_seq)           # [B, L, 256]

# Step 1.4: 融合位置信息
input_emb = item_emb + position_embedding          # [B, L, 256]
input_emb = self.LayerNorm(input_emb)
input_emb = self.dropout(input_emb)

# Step 1.5: Transformer Encoder（SASRec核心）
extended_attention_mask = self.get_attention_mask(item_seq)  # Causal mask
trm_output = self.trm_encoder(
    input_emb, 
    extended_attention_mask,
    output_all_encoded_layers=True
)  # List of [B, L, 256]

# Step 1.6: 获取序列表示（取最后一个有效位置）
output = trm_output[-1]                            # [B, L, 256]
seq_output = self.gather_indexes(output, item_seq_len - 1)  # [B, 256]

# seq_output: 序列的隐状态表示（纯ID特征）
```

**关键参数：**
- `hidden_size: 256` - ID embedding维度
- `n_layers: 2` - Transformer层数（默认）
- `n_heads: 2` - 注意力头数
- `token_dropout_prob: 0.2` - Token dropout概率

---

### 2. 文本特征路径（双路融合）

#### 2.1 加载与预处理

```python
# 代码位置：__init__() 方法，165-177行

# 加载两路特征
emb_base = load_npy(item_text_emb_path_base)  # TF-IDF [N, 256]
emb_llm = load_npy(item_text_emb_path_llm)    # Qwen3 [N, 256]

# L2 归一化（如果配置）
if self.normalize_text:
    emb_base = F.normalize(emb_base, p=2, dim=1)
    emb_llm = F.normalize(emb_llm, p=2, dim=1)

# 注册为buffer（冻结）
self.register_buffer("item_text_emb_base", emb_base)
self.register_buffer("item_text_emb_llm", emb_llm)
```

**配置参数：**
```yaml
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.base.npy
item_text_emb_path_llm: dataset/Amazon_Beauty/item_text_emb.qwen3.npy
normalize_text: true  # 启用L2归一化
```

#### 2.2 特征拼接

```python
# 代码位置：_gather_text_raw() 方法，688-696行

def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
    # ids_flat: [B]
    
    parts = []
    if self._text_mode in ("base", "both"):
        parts.append(self.item_text_emb_base[ids_flat])  # [B, 256]
    if self._text_mode in ("llm", "both"):
        parts.append(self.item_text_emb_llm[ids_flat])   # [B, 256]
    
    # 拼接两路特征
    return torch.cat(parts, dim=1)  # [B, 512]
```

**拼接策略：**
- `_text_mode = "both"` → base(256) + llm(256) = 512维
- `_text_mode = "base"` → 仅base(256)
- `_text_mode = "llm"` → 仅llm(256)

#### 2.3 文本投影网络（Cross + Deep）

```python
# 代码位置：_project_text() 方法，698-718行

def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
    # raw: [B, 512] (拼接后的文本特征)
    
    # Step 1: SENet Amplifier（可选增强）
    if self.text_amplifier is not None:
        raw = self.text_amplifier(raw)  # [B, 512]
    
    # Step 2: Cross Network（DCN-V2）
    cross_out = self.text_cross(raw)    # [B, 512]
    cross_out = self.text_cross_dropout(cross_out)
    
    # Step 3: Deep Network（MLP）
    deep_out = self.text_deep(raw)      # [B, 256]
    
    # Step 4: 拼接并投影到hidden_size
    fused = torch.cat([cross_out, deep_out], dim=1)  # [B, 768]
    proj = self.text_predictor(fused)   # Linear: [B, 768] -> [B, 256]
    
    # Step 5: LayerNorm归一化
    if self.text_proj_norm is not None:
        proj = self.text_proj_norm(proj)  # [B, 256]
    
    return proj  # 投影后的文本表示
```

**网络结构：**

**Cross Network (DCN-V2):**
```python
# 输入维度 = 512（base + llm）
DCNV2Cross(
    input_dim=512,
    num_layers=2  # text_cross_layer_num
)
# 输出: [B, 512]
```

**Deep Network (MLP):**
```python
MLPLayers(
    layers=[512, 256, 256],  # 3层MLP
    dropout=0.5,
    activation='relu',
    bn=text_mlp_bn  # 默认false
)
# 输出: [B, 256]
```

**Predictor (投影层):**
```python
nn.Linear(768, 256)  # 512(cross) + 256(deep) -> 256
```

**LayerNorm:**
```python
nn.LayerNorm(256)  # text_proj_norm
```

---

### 3. Item Embedding 融合（候选Item侧）

```python
# 代码位置：_get_fused_item_embeddings() 方法，741-862行

def _get_fused_item_embeddings(self, item_ids=None):
    # item_ids: [B] 或 None（全量）
    
    # Step 1: 获取Item ID Embedding
    if item_ids is None:
        all_ids = torch.arange(self.n_items)
        item_emb = self.item_embedding.weight  # [N, 256]
    else:
        all_ids = item_ids
        item_emb = self.item_embedding(item_ids)  # [B, 256]
    
    # Step 2: 获取文本特征
    text_raw = self._gather_text_raw(all_ids)  # [B, 512]
    if self.detach_text_emb:
        text_raw = text_raw.detach()  # 冻结文本梯度
    
    # Step 3: 计算有效文本权重（gate控制）
    alpha = torch.sigmoid(self.text_gate_param)  # 全局gate
    align_scale = 1.0 + self.alignment_weight    # 对齐scale
    temp_scale = 0.07 / self.temperature         # 温度scale
    effective_text_weight = alpha * self.text_weight * align_scale * temp_scale
    
    # Step 4: 应用per-item gate（可选）
    if self.text_item_gate_all is not None:
        gate = self.text_item_gate_all[all_ids]  # [B, 1]
        scaled_text = (effective_text_weight * gate) * text_raw
    else:
        scaled_text = effective_text_weight * text_raw  # [B, 512]
    
    # Step 5: SENet Amplifier（可选）
    if self.text_amplifier is not None:
        scaled_text = self.text_amplifier(scaled_text)
    
    # Step 6: Item Embedding归一化（可选）
    if self.item_emb_norm is not None:
        item_emb = self.item_emb_norm(item_emb)
    
    # Step 7: 拼接Item Emb和文本特征
    fusion_input = torch.cat([item_emb, scaled_text], dim=1)  # [B, 768]
    
    # Step 8: Cross Network融合
    cross_out = self.item_fusion_cross(fusion_input)  # [B, 768]
    cross_out = self.item_fusion_cross_dropout(cross_out)
    
    # Step 9: Deep Network融合
    deep_out = self.item_fusion_deep(fusion_input)  # [B, 256]
    
    # Step 10: 拼接并投影
    fused = torch.cat([cross_out, deep_out], dim=1)  # [B, 1024]
    fused_emb = self.item_fusion_predictor(fused)    # [B, 256]
    
    # Step 11: 最终LayerNorm
    if self.fused_item_norm is not None:
        fused_emb = self.fused_item_norm(fused_emb)  # [B, 256]
    
    return fused_emb  # 融合后的Item表示
```

**融合网络结构：**

**Item Fusion Cross Network:**
```python
DCNV2Cross(
    input_dim=768,  # 256(item) + 512(text)
    num_layers=2
)
# 输出: [B, 768]
```

**Item Fusion Deep Network:**
```python
MLPLayers(
    layers=[768, 256, 256],
    dropout=0.5
)
# 输出: [B, 256]
```

**Item Fusion Predictor:**
```python
nn.Linear(1024, 256)  # 768(cross) + 256(deep) -> 256
```

---

### 4. 打分与预测

#### 4.1 训练时（calculate_loss）

```python
# 代码位置：calculate_loss() 方法，914-999行

def calculate_loss(self, interaction):
    # Step 1: 获取序列表示
    seq_output = self.forward(item_seq, item_seq_len)  # [B, 256]
    
    # Step 2: 获取正样本Item Embedding（融合文本）
    pos_items = interaction[self.POS_ITEM_ID]
    pos_items_emb = self._get_fused_item_embeddings(pos_items)  # [B, 256]
    
    # Step 3: 计算正样本分数
    if self.loss_type == "BPR":
        # BPR损失
        neg_items = interaction[self.NEG_ITEM_ID]
        neg_items_emb = self._get_fused_item_embeddings(neg_items)
        pos_score = (seq_output * pos_items_emb).sum(dim=1)
        neg_score = (seq_output * neg_items_emb).sum(dim=1)
        loss = self.bpr_loss(pos_score, neg_score)
    else:
        # CE损失（全量候选）
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            pos_n = F.normalize(pos_items_emb, dim=1)
            pos_score = self.cosine_scale * (seq_n * pos_n).sum(dim=1)
        else:
            pos_score = (seq_output * pos_items_emb).sum(dim=1)
        
        # 全量Item Embedding
        all_items_emb = self._get_fused_item_embeddings()  # [N, 256]
        if self.cosine_score:
            all_n = F.normalize(all_items_emb, dim=1)
            logits = self.cosine_scale * torch.matmul(seq_n, all_n.T)
        else:
            logits = torch.matmul(seq_output, all_items_emb.T)
        
        loss = self.ce_loss(logits, pos_items)
    
    # Step 4: 对齐损失（可选）
    if self.alignment_weight > 0 and self.use_align:
        align_loss = self._info_nce_align(
            seq_output, 
            pos_items_emb
        )
        loss = loss + self.alignment_weight * align_loss
    
    return loss
```

#### 4.2 推理时（predict / full_sort_predict）

```python
# 代码位置：predict() 方法，1001-1014行

def predict(self, interaction):
    # Step 1: 获取序列表示
    seq_output = self.forward(item_seq, item_seq_len)  # [B, 256]
    
    # Step 2: 获取测试Item Embedding（融合文本）
    test_item = interaction[self.ITEM_ID]
    test_item_emb = self._get_fused_item_embeddings(test_item)  # [B, 256]
    
    # Step 3: 计算分数
    if self.cosine_score:
        seq_n = F.normalize(seq_output, dim=1)
        item_n = F.normalize(test_item_emb, dim=1)
        scores = self.cosine_scale * (seq_n * item_n).sum(dim=1)
    else:
        scores = (seq_output * test_item_emb).sum(dim=1)
    
    return scores  # [B]

def full_sort_predict(self, interaction):
    # 全量候选排序
    seq_output = self.forward(item_seq, item_seq_len)  # [B, 256]
    
    # 获取所有Item Embedding（融合文本）
    all_items_emb = self._get_fused_item_embeddings()  # [N, 256]
    
    if self.cosine_score:
        seq_n = F.normalize(seq_output, dim=1)
        item_n = F.normalize(all_items_emb, dim=1)
        scores = self.cosine_scale * torch.matmul(seq_n, item_n.T)
    else:
        scores = torch.matmul(seq_output, all_items_emb.T)
    
    return scores  # [B, N]
```

---

## 🎯 关键设计要点

### 1. 文本特征与ID特征的结合方式

**两种结合点：**

1. **候选Item侧（主要）：**
   - Item ID Embedding + 文本特征 → 融合后的Item表示
   - 通过 Cross Network + Deep Network 融合
   - 用于打分时的候选Item表示

2. **序列侧（可选）：**
   - `use_seq_text_cross: false`（当前配置未启用）
   - 如果启用，会在序列输出上再融合文本特征

**当前配置采用方式1（候选Item侧融合）**

---

### 2. 归一化策略

**多层归一化：**

1. **文本加载时L2归一化：**
   ```python
   if normalize_text:  # true
       emb_base = F.normalize(emb_base, p=2, dim=1)
       emb_llm = F.normalize(emb_llm, p=2, dim=1)
   ```

2. **文本投影后LayerNorm：**
   ```python
   if text_proj_norm:  # true
       text_proj = LayerNorm(text_proj)
   ```

3. **Item融合后LayerNorm：**
   ```python
   if fused_item_norm:  # true
       fused_emb = LayerNorm(fused_emb)
   ```

4. **打分前归一化（可选）：**
   ```python
   if cosine_score:  # true
       seq_n = F.normalize(seq_output, dim=1)
       item_n = F.normalize(test_item_emb, dim=1)
   ```

---

### 3. 门控机制

**多级门控：**

1. **全局文本gate：**
   ```python
   alpha = torch.sigmoid(self.text_gate_param)  # 可学习参数
   ```

2. **Per-item gate（可选）：**
   ```python
   gate = self.text_item_gate_all[item_ids]  # 每个item独立的gate
   ```

3. **有效权重计算：**
   ```python
   effective_text_weight = (
       alpha * 
       self.text_weight *          # 基础权重（0.8）
       (1 + alignment_weight) *    # 对齐scale
       (0.07 / temperature)        # 温度scale
   )
   ```

---

### 4. 损失函数

**主损失：**
- `loss_type: CE` - 交叉熵损失（全量候选）

**对齐损失（可选）：**
```python
if alignment_weight > 0:  # 0.05
    align_loss = InfoNCE(seq_output, pos_item_emb, temperature)
    total_loss = ce_loss + alignment_weight * align_loss
```

---

## 📊 参数配置总结

```yaml
# ID Embedding
hidden_size: 256           # 所有表示的统一维度

# 文本特征
item_text_emb_path_base: .../item_text_emb.base.npy  # TF-IDF
item_text_emb_path_llm: .../item_text_emb.qwen3.npy  # Qwen3
normalize_text: true       # L2归一化
detach_text_emb: true      # 冻结文本梯度

# 文本投影网络
use_cross: true            # 启用Cross Network
text_cross_layer_num: 2    # Cross层数
text_proj_norm: true       # 投影后LayerNorm

# Item融合
fuse_text_feature: true    # 启用文本融合
fused_item_norm: true      # 融合后LayerNorm
text_weight: 0.8           # 文本基础权重
text_gate_init: 0.5        # Gate初始值

# 对齐
use_align: true            # 启用对齐损失
alignment_weight: 0.05     # 对齐损失权重
temperature: 0.07          # InfoNCE温度

# 打分
cosine_score: true         # 使用余弦相似度
cosine_scale: 10.0         # 余弦scale
```

---

## 🔄 完整前向传播流程总结

```
输入: item_seq [B, L]
  ↓
[ID路径] 
  item_embedding → Transformer → seq_output [B, 256]
  
[文本路径]
  base_emb [N, 256] + llm_emb [N, 256] 
  → concat [N, 512]
  → Cross+Deep+Linear+LayerNorm 
  → text_proj [B, 256]
  
[Item融合]
  item_emb [B, 256] + text_features [B, 512]
  → concat [B, 768]
  → Cross+Deep+Linear+LayerNorm
  → fused_item_emb [B, 256]
  
[打分]
  score = cosine_scale × normalize(seq_output) · normalize(fused_item_emb)
  ↓
输出: scores [B] 或 [B, N]
```

---

**文档日期**: 2025-12-03  
**模型版本**: SASRecAlign (双路文本特征版本)  
**配置文件**: `sasrec_align_qwen3.yaml`

