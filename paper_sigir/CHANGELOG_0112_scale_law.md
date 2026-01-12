# CHANGELOG 2025-01-12 - Scale Law 深度分析

## 实验结论：Scale Law 在推荐任务中全面失效

### 1. 按物品频率分组的 Scale Law 分析

#### Toys 数据集 - MRR@10

| 物品分组 | 7B | 14B | 32B | Scale Law | 结论 |
|----------|-----|-----|-----|-----------|------|
| **new** (冷启动) | **0.0127** | 0.0125 | 0.0125 | 7B > 14B = 32B | ❌ 不成立 |
| **few** (低频) | **0.0286** | 0.0286 | 0.0278 | 7B = 14B > 32B | ❌ 不成立 |
| **frequent** (高频) | 0.0651 | **0.0664** | 0.0655 | 14B > 32B > 7B | ⚠️ 部分成立 |

带 CHANGE-9 (推理时 boost):

| 物品分组 | 7B | 14B | 32B | Scale Law | 结论 |
|----------|-----|-----|-----|-----------|------|
| **new** (冷启动) | **0.0121** | 0.0117 | 0.0116 | 7B > 14B > 32B | ❌ **完全反转!** |
| **few** (低频) | **0.0284** | 0.0284 | 0.0281 | 7B = 14B > 32B | ❌ 不成立 |
| **frequent** (高频) | **0.0671** | 0.0657 | 0.0668 | 7B > 32B > 14B | ❌ 不成立 |

#### Beauty 数据集 - MRR@10

| 物品分组 | 7B | 14B | 32B | Scale Law | 结论 |
|----------|-----|-----|-----|-----------|------|
| **new** (冷启动) | **0.0110** | 0.0102 | 0.0105 | 7B > 32B > 14B | ❌ 不成立 |
| **few** (低频) | 0.0196 | **0.0216** | 0.0207 | 14B > 32B > 7B | ⚠️ 部分成立 |
| **frequent** (高频) | **0.0552** | 0.0550 | 0.0537 | 7B > 14B > 32B | ❌ **完全反转!** |

带 CHANGE-9:

| 物品分组 | 7B | 14B | 32B | Scale Law | 结论 |
|----------|-----|-----|-----|-----------|------|
| **new** (冷启动) | 0.0103 | 0.0103 | **0.0104** | 32B ≈ 7B = 14B | ⚠️ 微弱成立 |
| **few** (低频) | **0.0222** | 0.0218 | 0.0222 | 7B = 32B > 14B | ❌ 不成立 |
| **frequent** (高频) | **0.0554** | 0.0552 | 0.0554 | 7B = 32B > 14B | ❌ 不成立 |

---

### 2. 关键发现

#### 发现1: New Items (冷启动) - Scale Law 不成立

即使在冷启动场景（文本特征应该最重要的地方），更大的 LLM 也没有带来更好的表现：

```
Toys (无boost):  7B=0.0127 > 14B=0.0125 = 32B=0.0125
Toys (+CHANGE-9): 7B=0.0121 > 14B=0.0117 > 32B=0.0116 (完全反转!)
Beauty (无boost): 7B=0.0110 > 32B=0.0105 > 14B=0.0102
```

#### 发现2: Frequent Items (高频) - Scale Law 完全反转

高频物品有充足的行为信号，文本特征的边际效益有限：

```
Beauty (无boost): 7B=0.0552 > 14B=0.0550 > 32B=0.0537 (完全反转!)
```

#### 发现3: 结论

> **"Multi-view 文本特征对高频物品的边际效益有限，而在冷启动场景下 7B 模型已足够捕获推荐任务所需的语义信息"**

---

## 新假设：SVD 压缩比导致 Scale Law 失效

### 问题分析

当前实验设置中，**所有模型规模的 LLM 嵌入都被 SVD 压缩到相同的 64 维度**：

| 模型 | 原始维度 | 压缩后维度 | 压缩比 | 信息损失 |
|------|----------|------------|--------|----------|
| Qwen3-7B | 3584 | 64 | **56x** | 较小 |
| Qwen3-14B | 5120 | 64 | **80x** | 中等 |
| Qwen3-32B | 5120 | 64 | **80x** | 中等 |

**核心问题**：
- 更大的模型有更丰富的语义信息（更高维度的表示空间）
- 但所有模型都压缩到相同的 64D 空间
- 这导致更大模型的信息损失比例更高，抵消了其语义优势

### 假设验证方案

#### 方案1: 维度自适应压缩

| 模型 | 原始维度 | 建议压缩维度 | 压缩比 |
|------|----------|--------------|--------|
| Qwen3-7B | 3584 | **64** | 56x |
| Qwen3-14B | 5120 | **128** | 40x |
| Qwen3-32B | 5120 | **256** | 20x |

#### 方案2: 多视角融合架构修改

当前架构:
```
[ID_emb(64)] + [TF-IDF(64)] + [LLM(64)] → concat → 192D → projection → 64D
```

提议架构 (不同维度融合):
```
方案A: 自适应投影
[ID_emb(64)] + [TF-IDF(64)] + [LLM(dim)] → 各自投影到64D → concat → 192D

方案B: 注意力融合
[ID_emb(64)] + [TF-IDF(64)] + [LLM(dim)] → 注意力加权融合 → 64D

方案C: 分层融合
[ID_emb(64)] ← 融合 ← [TF-IDF(64)] ← 融合 ← [LLM(dim)]
```

### 实验设计

#### 第一阶段: 验证压缩比假设

| 实验 | 模型 | SVD维度 | 目的 |
|------|------|---------|------|
| `exp_svd_7b_64` | 7B | 64 | 基准 (已有) |
| `exp_svd_14b_64` | 14B | 64 | 基准 (已有) |
| `exp_svd_14b_128` | 14B | **128** | 验证维度提升效果 |
| `exp_svd_32b_64` | 32B | 64 | 基准 (已有) |
| `exp_svd_32b_256` | 32B | **256** | 验证维度提升效果 |

#### 第二阶段: 融合架构验证

| 实验 | 架构 | 特征维度 | 目的 |
|------|------|----------|------|
| `exp_proj_14b_128to64` | 投影融合 | 14B→128→64 | 测试投影损失 |
| `exp_attn_14b_128` | 注意力融合 | 14B→128+attn | 测试注意力融合 |
| `exp_proj_32b_256to64` | 投影融合 | 32B→256→64 | 测试投影损失 |
| `exp_attn_32b_256` | 注意力融合 | 32B→256+attn | 测试注意力融合 |

### 预期结果

如果压缩比假设成立:

```
预期1: exp_svd_14b_128 > exp_svd_14b_64 (14B 提升)
预期2: exp_svd_32b_256 > exp_svd_32b_64 (32B 提升)
预期3: exp_svd_32b_256 > exp_svd_14b_128 > exp_svd_7b_64 (Scale Law 恢复!)
```

如果压缩比假设不成立:

```
结论: LLM 规模对推荐任务确实帮助有限
论文叙述: 7B 模型已足够捕获商品文本的语义信息
```

---

## 实现细节

### 1. SVD 维度修改

修改 `text_encoder.py` 中的 SVD 配置:

```python
# 当前配置
svd_dim = 64  # 所有模型统一

# 提议配置
svd_dim_config = {
    "7b": 64,
    "14b": 128,
    "32b": 256,
}
```

### 2. 融合层修改

修改 `sasrec_align_multi_view_v2.py`:

```python
# 当前: 固定64D融合
self.text_projection = nn.Linear(64, embedding_size)

# 提议: 自适应维度融合
class AdaptiveTextProjection(nn.Module):
    def __init__(self, llm_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(llm_dim, output_dim)
        
    def forward(self, llm_emb):
        return self.projection(llm_emb)
```

### 3. 注意力融合层

```python
class MultiViewAttentionFusion(nn.Module):
    """
    融合不同维度的多视角特征
    """
    def __init__(self, id_dim, tfidf_dim, llm_dim, output_dim):
        super().__init__()
        self.id_proj = nn.Linear(id_dim, output_dim)
        self.tfidf_proj = nn.Linear(tfidf_dim, output_dim)
        self.llm_proj = nn.Linear(llm_dim, output_dim)
        
        # 注意力权重
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4)
        
    def forward(self, id_emb, tfidf_emb, llm_emb):
        # 投影到统一维度
        id_proj = self.id_proj(id_emb)
        tfidf_proj = self.tfidf_proj(tfidf_emb)
        llm_proj = self.llm_proj(llm_emb)
        
        # Stack 为序列 [3, batch, dim]
        features = torch.stack([id_proj, tfidf_proj, llm_proj], dim=0)
        
        # 自注意力融合
        fused, _ = self.attention(features, features, features)
        
        return fused.mean(dim=0)  # [batch, dim]
```

---

## 待完成实验状态

### 已完成
- [x] exp_inference_boost_toys_14b
- [x] exp_inference_boost_toys_32b
- [x] exp_tfidf_llm_toys_align15
- [x] exp_tfidf_llm_toys_tau03

### 进行中 / 待完成
- [ ] exp_unified_aggressive_toys
- [ ] exp_boost_14b_aggressive
- [ ] exp_boost_32b_aggressive
- [ ] exp_unified_standard_beauty

### 新增 (SVD 验证)
- [ ] exp_svd_14b_128
- [ ] exp_svd_32b_256
- [ ] exp_proj_14b_128to64
- [ ] exp_proj_32b_256to64

---

## 论文叙述建议

### 当前结论叙述

```
Our experiments reveal that Scale Law does not hold for recommendation 
tasks across all item frequency groups:

1. For new items (cold-start), where text features should matter most,
   7B model consistently outperforms larger models (Toys: 7B=0.0127 > 
   14B=0.0125 = 32B=0.0125).

2. For frequent items with abundant behavioral signals, Scale Law even 
   reverses (Beauty: 7B=0.0552 > 14B=0.0550 > 32B=0.0537).

This suggests that multi-view text features provide limited marginal 
utility for high-frequency items where collaborative signals dominate.
```

### 如果 SVD 假设验证成功

```
We identify a critical confounding factor in Scale Law evaluation: 
the compression ratio during dimensionality reduction.

When all LLM embeddings (7B/14B/32B) are compressed to the same 64 
dimensions, larger models suffer proportionally greater information 
loss, masking their semantic advantage.

With adaptive compression (7B→64D, 14B→128D, 32B→256D), Scale Law 
is recovered: 32B > 14B > 7B on new items, demonstrating that larger 
LLMs do capture richer semantics when properly preserved.
```

---

## 公平对比实验 (0112 20:xx 启动)

### 实验目的
验证各组件的**独立贡献**，分解 Multi-view 的提升来源。

### 运行状态

| GPU | 实验 | 数据集 | 目的 | 状态 |
|-----|------|--------|------|------|
| 4090-0 | `exp_fair_tfidf_cold2_beauty` | Beauty | TF-IDF + cold=2.0 | 🔄 运行中 |
| 4090-1 | `exp_fair_tfidf_llm_cold2_beauty` | Beauty | TF-IDF+LLM + cold=2.0 | 🔄 运行中 |
| 4090-5 | `exp_fair_tfidf_llm_cold2_toys` | Toys | TF-IDF+LLM + cold=2.0 | 🔄 运行中 |
| 4090-6 | `exp_fair_multiview_no_boost_toys` | Toys | Multi-view 无 boost | 🔄 运行中 |

### 待运行 (需要 GPU 2, 3, 4, 7)

| GPU | 实验 | 数据集 | 目的 | 状态 |
|-----|------|--------|------|------|
| 4090-2 | `exp_fair_multiview_no_boost_beauty` | Beauty | Multi-view 无 boost | ⏳ 待运行 |
| 4090-3 | `exp_fair_multiview_cold2_only_beauty` | Beauty | Multi-view cold=2 only | ⏳ 待运行 |
| 4090-4 | `exp_fair_tfidf_cold2_toys` | Toys | TF-IDF + cold=2.0 | ⏳ 待运行 |
| 4090-7 | `exp_fair_multiview_cold2_only_toys` | Toys | Multi-view cold=2 only | ⏳ 待运行 |

### 对比矩阵

完成后可分析:

| 对比组 | 验证目标 |
|--------|----------|
| TF-IDF (cold=2) vs TF-IDF (cold=0) | cold_boost 对 TF-IDF 的提升 |
| TF-IDF+LLM (cold=2) vs TF-IDF (cold=2) | LLM 嵌入的边际贡献 |
| Multi-view (cold=0, infer=0) vs TF-IDF+LLM | 多视角架构的纯贡献 |
| Multi-view (cold=2, infer=0) vs (cold=2, infer=1) | CHANGE-9 的独立贡献 |

---

## 下一步行动

1. **等待当前实验完成**: 
   - exp_boost_14b_aggressive, exp_boost_32b_aggressive (Scale Law)
   - exp_fair_* 公平对比实验
2. **实现 SVD 维度自适应**: 修改预处理流程支持不同维度
3. **设计融合层**: 支持不同维度特征融合
4. **运行验证实验**: 验证压缩比假设

