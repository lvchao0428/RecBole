# 改进后的多视图模型架构

## 🎯 改进目标

**融合TF-IDF base特征 + 保持多视图处理 + 对齐双路模型的融合维度**

---

## 📊 改进后的完整架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      输入：用户序列                                │
│                  item_seq: [B, L]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   1. ID Embedding 路径                            │
│              （与原多视图模型完全相同）                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    seq_output [B, 256]


┌─────────────────────────────────────────────────────────────────┐
│              2. 混合文本特征路径（Base + Multi-View）               │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐          ┌─────────────────────────────┐
│   TF-IDF Base Path   │          │   Multi-View Qwen3 Path     │
└──────────────────────┘          └─────────────────────────────┘
         ↓                                     ↓
   base_emb [N, 256]           View_0 [N, 64] → Linear [256] → SENet
                               View_1 [N, 64] → Linear [256] → SENet
                               View_2 [N, 64] → Linear [256] → SENet
                               View_3 [N, 64] → Linear [256] → SENet
         ↓                                     ↓
    L2 Normalize                    Stack [B, 4, 256]
    (optional)                               ↓
         ↓                          Gate Weights × Views
    [B, 256]                                ↓
         │                          Concat [B, 1024]
         │                                  │
         │                                  │
         └──────────┬────────────────────────┘
                    ↓
        Concat [B, 1280]  (256 + 1024)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│            3. 混合特征投影（新增）                                  │
└─────────────────────────────────────────────────────────────────┘
                    ↓
          Linear Projection
           (1280 → 512)  ← 关键改进！
                    ↓
          text_proj [B, 512]
                    ↓
        (Optional) Normalize


┌─────────────────────────────────────────────────────────────────┐
│              4. Item Embedding 融合（与双路对齐）                   │
└─────────────────────────────────────────────────────────────────┘
                    ↓
        item_emb [B, 256]
                    ↓
        (Optional) LayerNorm
                    ↓
        ┌───────────┴───────────┐
        │                       │
    item_emb              text_proj
    [B, 256]              [B, 512]
        │                       ↓
        │                text_gate × weight
        │                       ↓
        │                 scaled_text
        │                  [B, 512]
        │                       │
        └──────────┬────────────┘
                   ↓
           Concat [B, 768]  ← 与双路模型对齐
                   ↓
      ┌────────────┴────────────┐
      │                         │
  Cross Network            Deep Network
   (DCN-V2, 2层)              (MLP)
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
                 ↓
       fused_item_emb
          [B, 256]


┌─────────────────────────────────────────────────────────────────┐
│           5. 多视图对齐损失（保持不变）                              │
└─────────────────────────────────────────────────────────────────┘

        id_item_emb [B, 256]
                   ↓
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
   InfoNCE_0   InfoNCE_1  InfoNCE_2  InfoNCE_3
        │          │          │          │
        └──────────┴──────────┴──────────┘
                   ↓
     Learnable Align Weights × Losses
                   ↓
        total_align_loss


┌─────────────────────────────────────────────────────────────────┐
│                    6. 打分与预测                                  │
└─────────────────────────────────────────────────────────────────┘

    seq_output [B, 256]  +  fused_item_emb [B, 256]
                   ↓
    score = cosine_scale × normalize(seq) · normalize(item)
```

---

## 🔑 关键改进点

### 改进1：融合TF-IDF Base特征

**改进前：**
```python
# 仅使用多视图
text_concat = concat([view_0, view_1, view_2, view_3])  # [B, 1024]
text_proj = Linear(1024, 256)(text_concat)              # [B, 256]
```

**改进后：**
```python
# 多视图 + Base特征
multiview_concat = concat([view_0, view_1, view_2, view_3])  # [B, 1024]
base_feat = base_emb[item_ids]                               # [B, 256]
text_concat = concat([base_feat, multiview_concat])          # [B, 1280]
text_proj = Linear(1280, 512)(text_concat)                   # [B, 512]
```

**优势：**
- ✅ 同时利用统计特征（TF-IDF）和语义特征（Qwen3）
- ✅ Base特征提供互补信息
- ✅ 与双路模型的设计理念一致

---

### 改进2：投影维度对齐

**改进前：**
```python
text_proj: [B, 256]
fusion_input: [B, 256 + 256 = 512]
Cross Network: 512² × 2层 (参数量: 0.52M)
```

**改进后：**
```python
text_proj: [B, 512]
fusion_input: [B, 256 + 512 = 768]
Cross Network: 768² × 2层 (参数量: 1.18M)
```

**优势：**
- ✅ 与双路模型融合维度对齐（768）
- ✅ 参数量相当（1.18M vs 1.18M）
- ✅ 信息损失从75%降到60%
- ✅ Cross Network建模能力提升

---

### 改进3：保持多视图优势

**保留的特性：**
- ✅ Per-View SENet增强
- ✅ Per-View Gate加权融合
- ✅ Per-View对齐损失（4个独立的InfoNCE）
- ✅ Learnable视图权重和对齐权重

**新增的特性：**
- ✅ TF-IDF base特征补充
- ✅ 更高维度的融合网络

---

## 📐 详细维度流

### 文本特征处理

```
┌─ Base Path ─┐                 ┌─ Multi-View Path ─┐
│             │                 │                    │
base_emb       View_0   View_1   View_2   View_3
[N, 256]       [N,64]   [N,64]   [N,64]   [N,64]
   ↓             ↓        ↓        ↓        ↓
Gather        Linear   Linear   Linear   Linear
   ↓           64→256   64→256   64→256   64→256
[B, 256]        ↓        ↓        ↓        ↓
   │          SENet    SENet    SENet    SENet
   │            ↓        ↓        ↓        ↓
   │         refined  refined  refined  refined
   │          [256]    [256]    [256]    [256]
   │            │        │        │        │
   │            └────────┴────────┴────────┘
   │                      ↓
   │              Gate加权 + Concat
   │                   [1024]
   │                      │
   └──────────┬───────────┘
              ↓
      Concat [1280]
              ↓
  Linear Projection
     (1280 → 512)
              ↓
      text_proj [512]
```

### Item融合处理

```
item_emb [256]  +  text_proj [512]
              ↓
      Concat [768]  ← 与双路模型对齐
              ↓
  ┌───────────┴───────────┐
  │                       │
Cross (768→768)      Deep (768→256)
  │                       │
  └───────────┬───────────┘
              ↓
      Concat [1024]
              ↓
  Linear (1024→256)
              ↓
     LayerNorm
              ↓
  fused_item_emb [256]
```

---

## 🔧 代码改动总结

### 1. 模型代码（sasrecalignmultiview.py）

**改动点1：投影维度调整**
```python
# Line 93-102 (修改后)
multiview_input_dim = self.num_text_views * self.hidden_size  # 1024
if self.item_text_emb_base is not None:
    base_dim = self.item_text_emb_base.shape[1]  # 256
    multiview_input_dim += base_dim  # 1280

self.multiview_concat_proj = nn.Linear(
    multiview_input_dim,      # 1280 (有base) 或 1024 (无base)
    self.hidden_size * 2      # 512
)
```

**改动点2：拼接base特征**
```python
# Line 219-230 (新增)
# Concat with base text features if available
if self.item_text_emb_base is not None:
    base_feat = self.item_text_emb_base[all_ids]  # [B, 256]
    if self.detach_text_emb:
        base_feat = base_feat.detach()
    text_concat = torch.cat([base_feat, text_concat], dim=-1)  # [B, 1280]
```

**改动点3：重新初始化融合网络**
```python
# Line 103-125 (新增)
fusion_input_dim = self.hidden_size + self.hidden_size * 2  # 768

self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=2)
self.item_fusion_deep = MLPLayers([fusion_input_dim, self.hidden_size])
self.item_fusion_predictor = nn.Linear(
    fusion_input_dim + self.hidden_size,  # 1024
    self.hidden_size
)
```

---

### 2. 配置文件（sasrec_align_multi_view.yaml）

**保持不变，base路径已配置：**
```yaml
item_text_emb_path_base: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy
item_text_emb_split_dir: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split
```

---

## 📊 改进前后对比

| 维度 | 改进前 | 改进后 | 双路模型（参考） |
|------|--------|--------|----------------|
| **Base特征** | ❌ 未使用 | ✅ 256维 | ✅ 256维 |
| **Multi-View特征** | 4×64=256 | 4×64=256 | ❌ 单视图256 |
| **视图处理后** | 1024维 | 1024维 | N/A |
| **拼接后** | 1024维 | **1280维** | 512维 |
| **投影后** | 256维 | **512维** ✅ | 512维 |
| **融合输入** | 512维 | **768维** ✅ | **768维** ✅ |
| **Cross参数量** | 0.52M | **1.18M** ✅ | 1.18M |
| **信息压缩率** | 75% | **60%** | 0% |

**关键改进：**
- ✅ 融合维度从512提升到768（与双路对齐）
- ✅ 同时利用统计特征和多视图语义特征
- ✅ 信息压缩从75%降到60%
- ✅ 参数量与双路模型相当

---

## 🎯 架构优势

### 相比原多视图模型

1. ✅ **信息容量提升**
   - 融合维度：512 → 768 (+50%)
   - Cross Network参数量翻倍

2. ✅ **特征互补**
   - TF-IDF：统计特征
   - Qwen3-4视图：多角度语义特征

3. ✅ **与双路模型可比性**
   - 相同的融合维度（768）
   - 公平的性能对比

---

### 相比双路模型

1. ✅ **多视图优势保留**
   - 4个语义视角
   - Per-View SENet增强
   - Per-View对齐

2. ✅ **特征丰富度更高**
   - Base + 4视图 vs Base + 单视图
   - 更全面的物品表示

3. ✅ **自适应能力更强**
   - Learnable视图权重
   - Learnable对齐权重

---

## 🔧 使用方法

### 1. 生成特征文件

```bash
# 生成TF-IDF base特征
bash tools/gen_tfidf_only_fast.sh

# 生成Qwen3多视图特征
bash tools/gen_qwen3_multiview_only.sh
```

**确保生成的文件：**
- `dataset/Amazon_Beauty/item_text_emb.base.npy`
- `dataset/Amazon_Beauty/qwen3_4views/view_0.npy`
- `dataset/Amazon_Beauty/qwen3_4views/view_1.npy`
- `dataset/Amazon_Beauty/qwen3_4views/view_2.npy`
- `dataset/Amazon_Beauty/qwen3_4views/view_3.npy`
- `dataset/Amazon_Beauty/qwen3_4views/views.json`

---

### 2. 配置文件

```yaml
# sasrec_align_multi_view.yaml

# 特征路径（两个都要配置）
item_text_emb_path_base: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy
item_text_emb_split_dir: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split

# 多视图配置
use_text_view_split: true
num_text_views: 4
text_view_senet_ratio: 4

# 融合配置
use_cross: true
text_cross_layer_num: 2
fuse_text_feature: true

# 归一化
normalize_text: true
text_proj_norm: true
fused_item_norm: true
```

---

### 3. 训练脚本

**直接使用现有脚本：**
```bash
bash two_phase_run_multiview_split.sh
```

**模型会自动检测：**
- ✅ 如果 `item_text_emb_path_base` 存在 → 融合base特征
- ✅ 如果不存在 → 仅使用多视图（向后兼容）

---

## 📊 预期性能提升

### 相比原多视图模型

| 指标 | 原模型 | 改进后 | 提升 |
|------|--------|--------|------|
| NDCG@10 | Baseline | Baseline + 3-5% | **+3-5%** |
| Recall@10 | Baseline | Baseline + 2-4% | **+2-4%** |

**提升来源：**
- ✅ TF-IDF base特征的互补信息
- ✅ 更高维度的融合网络
- ✅ 更强的建模能力

---

### 相比双路模型

**预期表现：**
- 🎯 **相当或更好**
- 原因：融合维度相同（768）+ 多视图优势

**可能的优势：**
- ✅ 多角度语义理解
- ✅ Per-View对齐更精细
- ✅ SENet自适应增强

---

## 🔍 向后兼容性

### 配置A：仅多视图（向后兼容）

```yaml
item_text_emb_path_base: ""  # 留空
item_text_emb_split_dir: /path/to/qwen3_4views_split
```

**行为：**
- 仅使用多视图特征
- 投影维度：1024 → 512
- 融合输入：256 + 512 = 768

---

### 配置B：多视图 + Base（新功能）

```yaml
item_text_emb_path_base: /path/to/item_text_emb.base.npy
item_text_emb_split_dir: /path/to/qwen3_4views_split
```

**行为：**
- 使用Base + 多视图
- 投影维度：1280 → 512
- 融合输入：256 + 512 = 768

---

### 配置C：双路模型（对照）

```yaml
item_text_emb_path_base: /path/to/item_text_emb.base.npy
item_text_emb_path_llm: /path/to/item_text_emb.qwen3.npy
use_text_view_split: false
```

**行为：**
- 使用Base + 单一LLM
- 拼接维度：512
- 融合输入：256 + 512 = 768

---

## 📝 代码变更详情

### 修改的文件

1. **`recbole/model/sequential_recommender/sasrecalignmultiview.py`**
   - ✅ 调整投影维度计算逻辑
   - ✅ 添加base特征拼接
   - ✅ 重新初始化融合网络
   - ✅ 更新注释和日志

2. **`sasrec_align_multi_view.yaml`**
   - ✅ 更新注释说明新架构

---

### 关键代码片段

**投影层初始化：**
```python
# 动态计算输入维度
multiview_input_dim = num_text_views * hidden_size  # 1024
if self.item_text_emb_base is not None:
    base_dim = self.item_text_emb_base.shape[1]  # 256
    multiview_input_dim += base_dim  # 1280

# 投影到512维
self.multiview_concat_proj = nn.Linear(multiview_input_dim, hidden_size * 2)
```

**特征拼接：**
```python
# 多视图处理
text_concat = concat([view_0, view_1, view_2, view_3])  # [1024]

# 如果有base特征，拼接
if self.item_text_emb_base is not None:
    base_feat = self.item_text_emb_base[item_ids]  # [256]
    text_concat = concat([base_feat, text_concat])  # [1280]

# 投影
text_proj = self.multiview_concat_proj(text_concat)  # [512]
```

**融合网络：**
```python
# 重新初始化，使用768维输入
fusion_input_dim = 256 + 512  # 768
self.item_fusion_cross = DCNV2Cross(768, num_layers=2)
self.item_fusion_deep = MLPLayers([768, 256])
self.item_fusion_predictor = nn.Linear(1024, 256)
```

---

## ✅ 验证清单

### 模型初始化时应该看到：

```
[INFO] SASRecAlignMultiView initialized: 4 views, SENet ratio=4, per-view alignment enabled
[INFO] Multi-view projection: [1280 → 512] | Base features: enabled | Fusion input dim: 768
```

**关键信息：**
- ✅ 投影输入：1280 (256 base + 1024 multi-view)
- ✅ 投影输出：512
- ✅ 融合输入：768 (256 item + 512 text)
- ✅ Base features: enabled

---

### 训练时检查：

```python
# 添加调试输出（可选）
print(f"base_feat shape: {base_feat.shape}")        # [B, 256]
print(f"multiview_concat shape: {text_concat.shape}")  # [B, 1280]
print(f"text_proj shape: {text_proj.shape}")        # [B, 512]
print(f"fusion_input shape: {fusion_input.shape}")  # [B, 768]
```

---

## 📚 完整架构文档

已创建：**`MULTIVIEW_IMPROVED_ARCHITECTURE.md`**

包含：
- 完整数据流图
- 详细维度分析
- 代码改动说明
- 使用方法
- 性能预期

---

## 🚀 下一步行动

### 1. 确认特征文件已生成

```bash
ls -lh dataset/Amazon_Beauty/item_text_emb.base.npy
ls -lh dataset/Amazon_Beauty/qwen3_4views/
```

### 2. 直接训练（配置已就绪）

```bash
bash two_phase_run_multiview_split.sh
```

**模型会自动：**
- ✅ 检测到base特征存在
- ✅ 使用1280→512的投影
- ✅ 融合维度为768

### 3. 观察日志确认

```bash
# 查看训练日志
tail -f run_metrics/watchdog_multiview_4views.log

# 应该看到：
# Multi-view projection: [1280 → 512] | Base features: enabled | Fusion input dim: 768
```

---

**改进完成！** 现在多视图模型已经：
- ✅ 融合了TF-IDF base特征
- ✅ 保持了原有的多视图处理（SENet、Gate、Per-View对齐）
- ✅ 对齐了双路模型的融合维度（768）
- ✅ 向后兼容（如果不设置base路径，仍然可以只用多视图）

需要我帮您验证实现是否正确吗？
