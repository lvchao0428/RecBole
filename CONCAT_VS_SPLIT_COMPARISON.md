# Multi-View Concat vs Split 模式对比

## 快速概览

| 特性 | **Concat 模式** | **Split 模式** |
|------|----------------|---------------|
| **脚本** | `two_phase_run_multiview_concat.sh` | `two_phase_run_multiview_split.sh` |
| **模型** | `SASRec_Align` | `SASRecAlignMultiView` |
| **特征文件** | `item_text_emb.qwen3.multiview.npy` | `qwen3_4views/` 目录 |
| **特征形式** | 单个 256 维向量 | 4 个 64 维向量 |
| **复杂度** | 简单 ⭐ | 复杂 |
| **灵活性** | 低 | 高 ⭐ |
| **推荐场景** | 快速实验、基线对比 | 最终模型、精调 |

## 详细对比

### 1. 特征结构

#### Concat 模式
```
item_text_emb.qwen3.multiview.npy  [N, 256]
  ↓ (已拼接)
  [view_0 | view_1 | view_2 | view_3]
  [  64   |   64   |   64   |   64  ] = 256 dims
```

- 特点：4 个视图已经拼接成单个向量
- 使用：作为整体输入模型
- 优势：简单直接，兼容单向量模型

#### Split 模式
```
qwen3_4views/
  ├── view_0.npy  [N, 64]  Identity
  ├── view_1.npy  [N, 64]  Function
  ├── view_2.npy  [N, 64]  Audience
  └── view_3.npy  [N, 64]  Category
```

- 特点：4 个视图分别存储
- 使用：每个视图独立处理
- 优势：可学习视图权重，更灵活

### 2. 模型架构

#### Concat 模式 (SASRec_Align)

```
Text Features (256-dim concat)
  ↓
Text MLP Projection
  ↓
Cross Network (2 layers)
  ↓
Gating Mechanism
  ↓
Fused with ID Embedding
```

**特点**：
- 将拼接向量作为整体处理
- 标准的文本对齐架构
- 较少的特殊处理

#### Split 模式 (SASRecAlignMultiView)

```
4 Individual Views (each 64-dim)
  ↓
Per-View SENet Enhancement
  ↓
Per-View Alignment Loss (learnable weights)
  ↓
Gated Fusion
  ↓
Concat → Projection → Cross Network
  ↓
Fused with ID Embedding
```

**特点**：
- 每个视图独立增强（SENet）
- 每个视图单独对齐
- 可学习的视图重要性权重
- 更复杂的融合机制

### 3. 关键参数对比

#### 共同参数（已对齐）

```bash
--phase_a_grid
--align_grid "0.01,0.03,0.05"
--tau_grid "0.05,0.07,0.1"
--backbone_burnin_epochs 10
--phase_a_epochs 6
--phase_a_eval_step 1
--phase_a_valid_metric "Recall@10"
--metric_baseline 0.0272
--lr_text_head 1e-3
--lr_dnn_cross 5e-4
--phase_b_epochs 40
--backbone_lr_scale 0.1
--seed 2025
```

#### 特有参数

**Concat 模式**：
```yaml
# sasrec_align_multiview_concat.yaml
item_text_emb_path_llm: .../item_text_emb.qwen3.multiview.npy
num_text_views: 4
text_use_senet: false  # 不使用 SENet
use_text_view_split: false  # 不使用分视图
```

**Split 模式**：
```yaml
# sasrec_align_multi_view.yaml
item_text_emb_split_dir: .../qwen3_4views
num_text_views: 4
text_view_senet_ratio: 4  # SENet 压缩比
use_text_view_split: true  # 使用分视图
text_view_half_precision: true
```

### 4. 计算复杂度

#### Concat 模式
- **内存占用**：低
- **计算量**：中等
- **训练速度**：快 ⭐

```
Forward Pass:
1. Text MLP: 256 → hidden
2. Cross Network: 2 layers
3. Gating: 1 scalar
4. Fusion: element-wise
```

#### Split 模式
- **内存占用**：中等（4 个视图分开处理）
- **计算量**：高（SENet + 独立对齐）
- **训练速度**：慢

```
Forward Pass (per item):
1. Per-view SENet: 4 × (64 → 16 → 64)
2. Per-view Alignment: 4 × alignment_loss
3. Gating: 1 scalar
4. Concat: 4 × 64 → 256
5. Projection + Cross: 256 → hidden
6. Fusion: element-wise
```

### 5. 性能预期

#### 理论分析

**Concat 优势**：
- ✅ 简单直接，少出错
- ✅ 训练速度快
- ✅ 适合快速实验

**Concat 劣势**：
- ❌ 无法学习视图重要性
- ❌ 所有视图同等对待
- ❌ 灵活性较低

**Split 优势**：
- ✅ 可学习视图权重（SENet）
- ✅ 每个视图独立优化
- ✅ 理论上性能上限更高

**Split 劣势**：
- ❌ 模型更复杂
- ❌ 训练时间更长
- ❌ 可能过拟合

#### 实验预期（Beauty 数据集）

| 模型 | Recall@10 | NDCG@10 | 训练时间 |
|------|-----------|---------|---------|
| ID-only Baseline | 0.0270 | 0.0165 | 1× |
| TF-IDF | 0.0285 | 0.0175 | 1.2× |
| **Concat (预期)** | **0.030-0.032** | **0.018-0.020** | **1.5×** |
| **Split (预期)** | **0.031-0.033** | **0.019-0.021** | **2×** |

*注：实际结果取决于超参数调优*

## 使用建议

### 何时使用 Concat 模式？

✅ **推荐场景**：
1. 快速验证多视图特征有效性
2. 资源受限（GPU 内存、时间）
3. 作为 baseline 对比
4. 初步超参数搜索

❌ **不推荐场景**：
1. 追求最佳性能
2. 需要可解释性（哪个视图重要）
3. 数据集特别复杂

### 何时使用 Split 模式？

✅ **推荐场景**：
1. 追求最佳性能
2. 充足的计算资源
3. 需要分析视图重要性
4. 最终模型部署

❌ **不推荐场景**：
1. 快速实验迭代
2. GPU 内存紧张
3. 时间有限

## 运行命令对比

### Concat 模式

```bash
# 1. 确保特征存在
ls -lh dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy

# 2. 运行训练
bash two_phase_run_multiview_concat.sh

# 3. 查看结果
ls -lh saved/phase_runs_multiview_concat/
```

### Split 模式

```bash
# 1. 确保特征存在
ls -lh dataset/Amazon_Beauty/qwen3_4views/

# 2. 运行训练
bash two_phase_run_multiview_split.sh

# 3. 查看结果
ls -lh saved/phase_runs_multiview_4views/
```

## 实验建议

### 完整对比实验

```bash
# 阶段 1：快速验证（Concat）
bash two_phase_run_multiview_concat.sh

# 等待完成，检查性能是否超过 baseline
# 如果 Recall@10 > 0.030，继续下一步

# 阶段 2：精细调优（Split）
bash two_phase_run_multiview_split.sh

# 对比两者结果
# 如果 Split 显著更好（+0.001），选择 Split
# 否则，Concat 已足够
```

### 并行对比

如果有多 GPU：

```bash
# GPU 0: Concat
CUDA_VISIBLE_DEVICES=0 bash two_phase_run_multiview_concat.sh &

# GPU 1: Split
CUDA_VISIBLE_DEVICES=1 bash two_phase_run_multiview_split.sh &

# 等待并对比
wait
```

## 代码差异关键点

### Concat 模式核心代码

```python
# 加载拼接的特征
text_emb = np.load('item_text_emb.qwen3.multiview.npy')  # [N, 256]

# 作为整体处理
text_feat = self.text_mlp(text_emb)  # 256 → hidden
text_feat = self.cross_network(text_feat)
gate = self.text_gate(...)
fused = id_emb + gate * text_feat
```

### Split 模式核心代码

```python
# 加载分离的视图
views = []
for i in range(4):
    view = np.load(f'qwen3_4views/view_{i}.npy')  # [N, 64]
    views.append(view)

# 每个视图独立处理
enhanced_views = []
for view in views:
    enhanced = self.senet[i](view)  # SENet 增强
    enhanced_views.append(enhanced)

# 拼接并融合
concat_feat = torch.cat(enhanced_views, dim=-1)  # [N, 256]
text_feat = self.projection(concat_feat)
text_feat = self.cross_network(text_feat)
gate = self.text_gate(...)
fused = id_emb + gate * text_feat
```

## 总结

| 维度 | Concat ⭐ 快速 | Split ⭐ 精细 |
|------|--------------|--------------|
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **性能** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可解释性** | ⭐⭐ | ⭐⭐⭐⭐ |

**推荐路径**：
1. 先用 **Concat** 快速验证
2. 如果效果好，再用 **Split** 精调
3. 最终根据性能/资源平衡选择

---

**当前状态**：两种模式都已更新并对齐最新特征结构，可以直接运行对比实验。

