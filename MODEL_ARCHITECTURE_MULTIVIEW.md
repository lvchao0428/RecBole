# SASRecAlignMultiView 完整模型架构详解

基于 `two_phase_run_multiview_split.sh` 的4视图多视角文本特征配置

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
│              （与SASRecAlign完全相同）                             │
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
                      LayerNorm → Dropout
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│            Transformer Encoder（序列建模）                         │
│         SASRec Transformer (self-attention layers)              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
                gather_indexes(seq_len - 1)
                           ↓
                    seq_output
                      [B, 256]  ← 序列表示（ID特征）


┌─────────────────────────────────────────────────────────────────┐
│           2. 多视图文本特征路径（4个独立视图）                        │
└─────────────────────────────────────────────────────────────────┘

  View 0          View 1          View 2          View 3
 (Identity)     (Function)      (Audience)      (Category)
     ↓               ↓               ↓               ↓
view_emb_0      view_emb_1      view_emb_2      view_emb_3
  [N, 64]         [N, 64]         [N, 64]         [N, 64]
     ↓               ↓               ↓               ↓
  Linear          Linear          Linear          Linear
  64→256          64→256          64→256          64→256
     ↓               ↓               ↓               ↓
 projected_0     projected_1     projected_2     projected_3
  [B, 256]        [B, 256]        [B, 256]        [B, 256]
     ↓               ↓               ↓               ↓
┌────────────────────────────────────────────────────────────────┐
│            Per-View SENet Enhancement（注意力增强）               │
└────────────────────────────────────────────────────────────────┘
     ↓               ↓               ↓               ↓
  SENet_0         SENet_1         SENet_2         SENet_3
  (256→64         (256→64         (256→64         (256→64
   →256)           →256)           →256)           →256)
     ↓               ↓               ↓               ↓
excitation_0    excitation_1    excitation_2    excitation_3
  [B, 256]        [B, 256]        [B, 256]        [B, 256]
     ↓               ↓               ↓               ↓
  refined_0       refined_1       refined_2       refined_3
  [B, 256]        [B, 256]        [B, 256]        [B, 256]
     ↓               ↓               ↓               ↓
     └───────────────┴───────────────┴───────────────┘
                          ↓
                Stack → [B, 4, 256]
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│              3. 视图门控融合（Learnable Weights）                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
         text_view_gate_params (learnable) [4]
                  ↓ sigmoid + normalize
         gate_weights = [w0, w1, w2, w3]
                          ↓
         ┌────────────────┼────────────────┐
         │                │                │
    w0×view_0        w1×view_1   ...   w3×view_3
    [B, 256]         [B, 256]         [B, 256]
         │                │                │
         └────────────────┴────────────────┘
                          ↓
              Concat [B, 1024]  (4×256)
                          ↓
           Linear Projection
            (1024 → 256)
                          ↓
              text_proj [B, 256]
                          ↓
            (Optional) Normalize


┌─────────────────────────────────────────────────────────────────┐
│              4. Item Embedding 融合（候选Item）                    │
└─────────────────────────────────────────────────────────────────┘

      对于候选item_ids:
                   ↓
            item_embedding
              [B, 256]
                   ↓
          (Optional) LayerNorm
                   ↓
        ┌──────────┴──────────┐
        │                     │
    item_emb            text_proj
    [B, 256]             [B, 256]
        │                     ↓
        │              text_gate × weight
        │                     ↓
        │              scaled_text
        │               [B, 256]
        │                     │
        └──────────┬───────────┘
                   ↓
           Concat [B, 512]
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2)                   (MLP)
   [B, 512]                 [B, 256]
      │                         │
  Dropout                       │
      │                         │
      └──────────┬──────────────┘
                 ↓
          Concat [B, 768]
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
│                 5. 多视图对齐损失（训练时）                         │
└─────────────────────────────────────────────────────────────────┘

        id_item_emb [B, 256]  (正样本的ID embedding)
                   ↓
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
   InfoNCE_0   InfoNCE_1  InfoNCE_2  InfoNCE_3
   (id, v0)    (id, v1)   (id, v2)   (id, v3)
        │          │          │          │
   align_loss_0 align_loss_1 ... align_loss_3
        │          │          │          │
        └──────────┴──────────┴──────────┘
                   ↓
     text_view_align_weights (learnable) [4]
              ↓ softmax
     [aw0, aw1, aw2, aw3]  (normalized)
                   ↓
     total_align_loss = Σ(awi × align_loss_i)
                   ↓
     total_loss = ce_loss + alignment_weight × total_align_loss


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
                   ↓
              [B] scores
```

---

## 🔧 详细模块说明

### 1. ID Embedding 路径（与SASRecAlign相同）

序列建模部分完全继承自父类 `SASRecAlign`，包括：
- Token Dropout
- Position Embedding
- Transformer Encoder
- 序列表示提取

**详细流程见：** `MODEL_ARCHITECTURE_DETAILED.md` 第1节

---

### 2. 多视图文本特征路径（4个独立视图）

#### 2.1 加载分视图特征

```python
# 代码位置：__init__() 方法，44-92行

# 读取视图元数据
meta_path = "dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split/views.json"
meta = json.load(meta_path)

# views.json 示例：
{
  "num_items": 259205,
  "num_prompts": 4,
  "prompts": [
    {
      "index": 0,
      "prompt": "Identify the item: [TITLE] {text}",
      "file": "view_0.npy",
      "vector_dim": 64
    },
    {
      "index": 1,
      "prompt": "What are the main functions and features of [TITLE] {text}?",
      "file": "view_1.npy",
      "vector_dim": 64
    },
    {
      "index": 2,
      "prompt": "Who is the target audience or user group for [TITLE] {text}?",
      "file": "view_2.npy",
      "vector_dim": 64
    },
    {
      "index": 3,
      "prompt": "Categorize the item [TITLE] {text} and describe its context.",
      "file": "view_3.npy",
      "vector_dim": 64
    }
  ]
}

# 为每个视图创建：
for view in prompts:
    # 1. 加载npy文件
    view_emb = np.load(f"view_{idx}.npy")  # [N, 64]
    self.register_buffer(f"text_view_emb_{idx}", view_emb)
    
    # 2. 创建投影层：64 → 256
    self.text_view_proj[idx] = nn.Linear(64, 256)
    
    # 3. 创建SENet模块
    self.text_view_senet[idx] = SENet(256, reduction=4)
```

**配置参数：**
```yaml
item_text_emb_split_dir: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split
use_text_view_split: true
num_text_views: 4
text_view_senet_ratio: 4  # SENet压缩比
text_view_half_precision: true  # 使用float16存储
```

---

#### 2.2 视图特征提取与增强

```python
# 代码位置：_gather_text_views() 方法，112-146行

def _gather_text_views(self, ids_flat: torch.Tensor) -> torch.Tensor:
    # ids_flat: [B]
    
    view_features = []
    for idx in range(4):  # 4个视图
        # Step 1: 从buffer加载原始视图embedding
        view_emb_table = self.text_view_emb_{idx}  # [N, 64]
        gathered = view_emb_table[ids_flat]  # [B, 64]
        
        # Step 2: 投影到hidden_size (256)
        projected = self.text_view_proj[idx](gathered)  # [B, 256]
        
        # Step 3: SENet增强（Squeeze-and-Excitation）
        excitation = self.text_view_senet[idx](projected)  # [B, 256]
        refined = projected * excitation  # 逐元素相乘 [B, 256]
        
        view_features.append(refined)
    
    # Step 4: 堆叠所有视图
    stacked = torch.stack(view_features, dim=1)  # [B, 4, 256]
    return stacked
```

**SENet 模块结构：**

```python
# 每个视图独立的SENet
SENet(
    nn.Linear(256, 64),   # Squeeze: 256 → 64 (ratio=4)
    nn.ReLU(),
    nn.Linear(64, 256),   # Excitation: 64 → 256
    nn.Sigmoid()          # 生成attention权重 ∈ [0,1]
)
# 输出: excitation [B, 256]
# 应用: refined = projected × excitation
```

**SENet作用：**
- 自适应地调整每个通道（特征维度）的重要性
- 类似于"特征级别的注意力机制"
- 增强有用的特征，抑制无用的特征

---

#### 2.3 视图门控融合

```python
# 代码位置：_get_fused_item_embeddings() 方法，184-209行

# Step 1: 获取SENet增强后的视图特征
view_stack = self._gather_text_views(all_ids)  # [B, 4, 256]

if self.detach_text_emb:
    view_stack = view_stack.detach()  # 冻结文本梯度

# Step 2: 计算视图权重（可学习参数）
view_weights = torch.sigmoid(self.text_view_gate_params)  # [4]
view_weights = view_weights / view_weights.sum()  # 归一化 [4]

# Step 3: 加权每个视图并拼接
weighted_views = []
for idx in range(4):
    view_feat = view_stack[:, idx, :]  # [B, 256]
    weighted = view_weights[idx] * view_feat  # 标量权重
    weighted_views.append(weighted)

# 拼接：4个[B, 256] → [B, 1024]
text_concat = torch.cat(weighted_views, dim=-1)

# Step 4: 投影到hidden_size
text_proj = self.multiview_concat_proj(text_concat)  # [B, 1024] → [B, 256]
```

**可学习门控参数：**
```python
# 初始化（__init__）
self.text_view_gate_params = nn.Parameter(torch.zeros(4))

# 前向时动态计算权重
gate_weights = sigmoid(params) / sigmoid(params).sum()
# 示例输出：[0.22, 0.28, 0.25, 0.25] (归一化后和为1.0)
```

**投影层：**
```python
nn.Linear(1024, 256)  # num_views * hidden_size → hidden_size
```

---

### 3. Item Embedding 融合（与父类类似）

```python
# 代码位置：_fuse_with_cross_network() 方法，211-288行

def _fuse_with_cross_network(item_emb, text_proj, item_ids):
    # item_emb: [B, 256]
    # text_proj: [B, 256] (多视图融合后)
    
    # Step 1: 文本归一化（可选）
    if self.normalize_text:
        text_proj = F.normalize(text_proj, dim=1)
    
    # Step 2: 计算有效文本权重
    alpha = torch.sigmoid(self.text_gate_param)  # 全局gate
    effective_text_weight = alpha * self.text_weight * temp_scale
    
    # Step 3: Per-item gate（可选）
    if self.text_item_gate_all is not None:
        gate = self.text_item_gate_all[item_ids]
        effective_text_weight = effective_text_weight * gate
    
    scaled_text = effective_text_weight * text_proj  # [B, 256]
    
    # Step 4: Item Embedding归一化（可选）
    if self.item_emb_norm is not None:
        item_emb = self.item_emb_norm(item_emb)
    
    # Step 5: Cross Network融合
    fusion_input = torch.cat([item_emb, scaled_text], dim=1)  # [B, 512]
    
    cross_out = self.item_fusion_cross(fusion_input)  # [B, 512]
    cross_out = self.item_fusion_cross_dropout(cross_out)
    deep_out = self.item_fusion_deep(fusion_input)    # [B, 256]
    
    fused = torch.cat([cross_out, deep_out], dim=1)   # [B, 768]
    fused_emb = self.item_fusion_predictor(fused)     # [B, 256]
    
    # Step 6: 最终归一化
    if self.fused_item_norm is not None:
        fused_emb = self.fused_item_norm(fused_emb)
    
    return fused_emb
```

**融合网络结构：**

```
输入: item_emb[256] + text_proj[256] = [512]
  ↓
Cross Network (DCN-V2, 2层): [512] → [512]
Deep Network (MLP): [512] → [256]
  ↓
Concat: [512] + [256] = [768]
  ↓
Linear: [768] → [256]
  ↓
LayerNorm: [256]
  ↓
输出: fused_item_emb [256]
```

---

### 4. 多视图对齐损失（训练时）

```python
# 代码位置：calculate_loss() 方法，291-355行

def calculate_loss(self, interaction):
    # Step 1: 计算基础损失（CE/BPR）
    loss = super().calculate_loss(interaction)  # 调用父类
    
    # Step 2: 添加多视图对齐损失
    if self.use_align and self.alignment_weight > 0:
        pos_items = interaction[self.POS_ITEM_ID]
        
        # 获取ID embedding
        id_item_emb = self.item_embedding(pos_items)  # [B, 256]
        
        # 获取所有视图特征
        view_stack = self._gather_text_views(pos_items)  # [B, 4, 256]
        if self.detach_text_emb:
            view_stack = view_stack.detach()
        
        # 为每个视图计算独立的对齐损失
        per_view_align_losses = []
        for idx in range(4):
            view_feat = view_stack[:, idx, :]  # [B, 256]
            
            # InfoNCE对齐损失
            align_loss_i = self._info_nce_align(id_item_emb, view_feat)
            per_view_align_losses.append(align_loss_i)
        
        # 应用可学习的视图对齐权重
        align_weights = F.softmax(self.text_view_align_weights, dim=0)  # [4]
        
        # 加权求和
        total_align_loss = sum(
            align_weights[idx] * per_view_align_losses[idx]
            for idx in range(4)
        )
        
        # 添加到总损失
        loss = loss + self.alignment_weight * total_align_loss
    
    return loss
```

**可学习对齐权重：**
```python
# 初始化（__init__）
self.text_view_align_weights = nn.Parameter(torch.ones(4))

# 训练时动态学习每个视图的对齐重要性
align_weights = F.softmax(params, dim=0)
# 示例：[0.30, 0.25, 0.20, 0.25] (和为1.0)
```

**InfoNCE对齐损失：**
```python
def _info_nce_align(a: Tensor, b: Tensor) -> Tensor:
    # a: ID embedding [B, 256]
    # b: 视图文本特征 [B, 256]
    # temperature: 0.07
    
    # 归一化
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(a, b.T) / temperature  # [B, B]
    
    # 对角线是正样本，其余是负样本
    labels = torch.arange(B, device=a.device)
    
    # 交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

---

### 5. 打分与预测（与父类相同）

```python
# predict() 和 full_sort_predict() 继承自 SASRecAlign
# 详细流程见 MODEL_ARCHITECTURE_DETAILED.md 第6节

def full_sort_predict(self, interaction):
    seq_output = self.forward(item_seq, item_seq_len)  # [B, 256]
    
    # 获取所有Item融合后的embedding（包含多视图文本特征）
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

### 1. 多视图设计哲学

**4个视图对应不同语义角度：**

| 视图索引 | 语义角度 | 提示词模板 | 特征维度 |
|---------|---------|-----------|---------|
| **View 0** | Identity（身份） | "Identify the item: [TITLE] {text}" | 64 |
| **View 1** | Function（功能） | "What are the main functions and features..." | 64 |
| **View 2** | Audience（受众） | "Who is the target audience..." | 64 |
| **View 3** | Category（类别） | "Categorize the item..." | 64 |

**设计优势：**
- ✅ 捕捉物品的多个方面信息
- ✅ 不同视图相互补充，减少信息偏差
- ✅ 每个视图独立训练和优化
- ✅ 可学习的视图重要性权重

---

### 2. 三级门控机制

**Level 1: Per-View Gate（视图级门控）**
```python
view_weights = sigmoid(text_view_gate_params)  # [4]
view_weights = view_weights / sum(view_weights)
# 控制每个视图的全局重要性
```

**Level 2: Global Text Gate（文本级门控）**
```python
alpha = sigmoid(text_gate_param)  # 标量
effective_weight = alpha * text_weight
# 控制文本特征的整体重要性
```

**Level 3: Per-Item Gate（物品级门控，可选）**
```python
gate = text_item_gate_all[item_ids]  # [B, 1]
effective_weight = effective_weight * gate
# 每个物品独立的文本权重
```

---

### 3. 双重对齐策略

**Per-View Alignment（每个视图独立对齐）：**
```python
for view_idx in range(4):
    view_feat = view_stack[:, view_idx, :]
    align_loss_i = InfoNCE(id_emb, view_feat)
    
total_align_loss = Σ(align_weights[i] × align_loss_i)
```

**优势：**
- ✅ 每个视图都与ID embedding对齐
- ✅ 可学习的对齐权重（`text_view_align_weights`）
- ✅ 避免某些视图被忽略
- ✅ 促进多视图一致性

---

### 4. SENet增强机制

**每个视图独立的SENet：**

```python
# 对于每个视图 i
projected = Linear(view_emb[i])  # [B, 256]

# SENet: 自适应特征重加权
squeeze = Linear(projected, 64)   # 压缩
excitation = Sigmoid(Linear(squeeze, 256))  # 激活
refined = projected × excitation   # 逐元素相乘

# excitation ∈ [0, 1]^256
# 每个特征维度有自己的权重
```

**作用：**
- 强化重要特征维度
- 抑制噪声特征维度
- 增加特征表达能力

---

### 5. 归一化策略（4层）

1. **视图投影后（可选）：** 无归一化，保留视图间差异
2. **视图融合后：** 可选L2归一化（`normalize_text: true`）
3. **Item融合后：** LayerNorm（`fused_item_norm: true`）
4. **打分前：** L2归一化（`cosine_score: true`）

---

## 📐 网络结构细节

### Per-View Projection + SENet

```
输入 (每个视图): view_emb [N, 64]
  ↓
Linear Projection: [64] → [256]
  ↓
SENet Enhancement:
  ├─ Squeeze: [256] → [64] (Linear + ReLU)
  ├─ Excitation: [64] → [256] (Linear + Sigmoid)
  └─ Multiply: projected × excitation
  ↓
输出: refined_view [B, 256]
```

### Multi-View Concat Projection

```
输入: 4个视图 [B, 4, 256]
  ↓
Per-View Gating:
  view_weights = softmax(sigmoid(gate_params))  # [4]
  weighted_views = [w_i × view_i for i in range(4)]
  ↓
Concat: [B, 256] × 4 → [B, 1024]
  ↓
Linear Projection: [1024] → [256]
  ↓
(Optional) Normalize
  ↓
输出: text_proj [256]
```

### Item Fusion Network (继承自父类)

```
输入: item_emb[256] + text_proj[256]
  ↓
(Optional) item_emb_norm: LayerNorm
  ↓
Concat: [512]
  ↓
├─ Cross Network (DCN-V2, 2层): [512] → [512]
│  └─ Dropout
└─ Deep Network (MLP): [512] → [256]
  ↓
Concat: [768]
  ↓
Linear: [768] → [256]
  ↓
LayerNorm (fused_item_norm)
  ↓
输出: fused_item_emb [256]
```

---

## 📊 参数配置总结

```yaml
# 多视图配置
item_text_emb_split_dir: .../item_text_emb_qwen3_4views_split
use_text_view_split: true
num_text_views: 4
text_view_senet_ratio: 4       # SENet压缩比
text_view_half_precision: true # float16存储

# Cross Network融合
use_cross: true                # 启用Cross Network
text_cross_layer_num: 2        # Cross层数

# 对齐
use_align: true                # 启用对齐损失
alignment_weight: 0.05         # 对齐损失权重
temperature: 0.07              # InfoNCE温度（共享）

# 归一化
normalize_text: true           # 文本L2归一化
text_proj_norm: true           # 投影后LayerNorm
fused_item_norm: true          # 融合后LayerNorm

# 文本权重
text_weight: 0.8               # 基础文本权重
text_gate_init: 0.5            # Gate初始值

# 打分
cosine_score: true             # 余弦相似度
cosine_scale: 10.0             # 余弦scale

# 正则化
text_gate_reg_l2: 0.05         # Gate的L2正则
```

---

## 🔄 完整前向传播流程总结

```
输入: item_seq [B, L]
  ↓
┌─────────────────────────────────────────┐
│ [ID路径]                                 │
│ item_embedding → Transformer            │
│ → seq_output [B, 256]                   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ [多视图路径]                              │
│ View_0[64] → Proj[256] → SENet          │
│ View_1[64] → Proj[256] → SENet          │
│ View_2[64] → Proj[256] → SENet          │
│ View_3[64] → Proj[256] → SENet          │
│   ↓                                     │
│ Stack [B, 4, 256]                       │
│   ↓                                     │
│ Gate Weights × Views                    │
│   ↓                                     │
│ Concat [B, 1024]                        │
│   ↓                                     │
│ Linear Proj [B, 256]                    │
│ → text_proj [B, 256]                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ [Item融合]                               │
│ item_emb[256] + text_proj[256]          │
│   ↓                                     │
│ Concat [B, 512]                         │
│   ↓                                     │
│ Cross Network + Deep Network            │
│   ↓                                     │
│ Linear + LayerNorm                      │
│ → fused_item_emb [B, 256]               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ [对齐损失] (训练时)                       │
│ For each view i:                        │
│   align_loss_i = InfoNCE(id_emb, view_i)│
│ total_align = Σ(align_weights[i] × L_i)│
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ [打分]                                   │
│ score = cosine_scale ×                  │
│   normalize(seq_output) ·               │
│   normalize(fused_item_emb)             │
└─────────────────────────────────────────┘
  ↓
输出: scores [B] 或 [B, N]
```

---

## 🆚 与双路模型（SASRecAlign）对比

| 特性 | SASRecAlign (双路) | SASRecAlignMultiView (多视图) |
|------|-------------------|------------------------------|
| **文本特征** | Base[256] + LLM[256] = 512 | View_0~3[64×4] = 256 |
| **特征增强** | 无 | Per-View SENet |
| **视图融合** | 简单Concat | Learnable Weighted Concat |
| **对齐策略** | 单一对齐损失 | Per-View对齐 + Learnable Weights |
| **门控机制** | Global Gate | Per-View + Global + Per-Item |
| **投影网络** | Cross + Deep | View Proj → Multi-View Proj → Cross + Deep |
| **参数量** | 较少 | 较多（4个SENet + 4个Proj） |
| **表达能力** | 单一视角 | 多视角互补 |

---

## 🎯 多视图模型的优势

### 1. 语义多样性
- 4个视图从不同角度描述物品
- 减少单一描述的偏差
- 更全面的物品表示

### 2. 自适应融合
- 可学习的视图权重
- 自动发现重要视图
- 动态调整视图贡献

### 3. 特征增强
- SENet自适应特征重加权
- 强化有用信息，抑制噪声
- 提升特征表达能力

### 4. 鲁棒性
- 即使某个视图质量差，其他视图可补偿
- Per-view对齐确保每个视图都有用
- 降低对单一特征源的依赖

---

## 📝 关键可学习参数

### 1. 视图融合权重
```python
text_view_gate_params: [4]  # 每个视图的重要性
# 训练后可能的分布：[0.22, 0.28, 0.25, 0.25]
```

### 2. 视图对齐权重
```python
text_view_align_weights: [4]  # 每个视图对齐的权重
# 训练后可能的分布：[0.30, 0.25, 0.20, 0.25]
```

### 3. SENet参数（每个视图独立）
```python
for i in range(4):
    SENet_i: Linear(256, 64) + Linear(64, 256)
```

### 4. 投影参数
```python
text_view_proj[i]: Linear(64, 256)  # 4个独立投影
multiview_concat_proj: Linear(1024, 256)  # 多视图拼接投影
```

---

**文档日期**: 2025-12-03  
**模型版本**: SASRecAlignMultiView (4视图版本)  
**配置文件**: `sasrec_align_multi_view.yaml`  
**特征来源**: 4-view Qwen3 split embeddings (Identity, Function, Audience, Category)

