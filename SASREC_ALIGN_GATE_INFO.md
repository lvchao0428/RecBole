# SASRecAlign 模型的 GATE 机制说明

## ✅ 已有的 GATE 机制

SASRecAlign 模型**已经包含了完整的 GATE 机制**，而且是**双层设计**。

---

## 🔍 GATE 实现详情

### 1. 全局可学习 Gate (`text_gate_param`)

**代码位置**: `recbole/model/sequential_recommender/sasrec_align.py` Line 128

```python
# Line 124-128
self.text_gate_init = float(config["text_gate_init"]) if "text_gate_init" in config else 0.5
self.text_gate_reg_l2 = float(config["text_gate_reg_l2"]) if "text_gate_reg_l2" in config else 0.0
self.text_gate_reg_entropy = float(config["text_gate_reg_entropy"]) if "text_gate_reg_entropy" in config else 0.0
# learnable global gate alpha in [0,1] via sigmoid
self.text_gate_param = nn.Parameter(torch.tensor(self.text_gate_init, dtype=torch.float32))
```

**特点**:
- ✅ **可学习参数**: 通过梯度下降自动优化
- ✅ **初始值可配置**: `text_gate_init` (默认 0.5)
- ✅ **L2 正则化**: `text_gate_reg_l2` (默认 0.05)
- ✅ **熵正则化**: `text_gate_reg_entropy` (默认 0.0)
- ✅ **范围 [0,1]**: 通过 `torch.sigmoid()` 转换

**使用位置**: Line 802

```python
alpha = torch.sigmoid(self.text_gate_param)
# ... 计算 effective_text_weight
effective_text_weight = alpha * self.text_weight * align_scale * temp_scale
```

---

### 2. Per-Item Gate (`text_item_gate_all`)

**代码位置**: Line 235-247

```python
# Per-item gating based on item interaction frequency
if self.text_tail_threshold > 0 and self.item_text_emb_base is not None:
    item_counts = np.zeros(self.n_items, dtype=np.float32)
    inter_feat = dataset.inter_feat
    for iid in inter_feat[self.ITEM_ID]:
        item_counts[iid] += 1.0
    gate = (item_counts <= self.text_tail_threshold).astype(np.float32)
    gate = torch.from_numpy(gate)
else:
    gate = None
# shape [n_items], 1.0 means enable text for that item
self.register_buffer("text_item_gate_all", gate)
```

**特点**:
- ✅ **基于交互次数**: 区分热门 item 和冷启动 item
- ✅ **阈值可配置**: `text_tail_threshold` (默认 5)
- ✅ **二值化 gate**: 交互次数 ≤ 阈值的 item，gate=1.0（启用文本）
- ✅ **Buffer 存储**: 不参与梯度计算，固定值

**使用位置**: Line 642-645, 817-825

```python
# 获取 per-item gate
if self.text_item_gate_all is not None:
    gate = self.text_item_gate_all[all_ids].to(device).unsqueeze(1)
else:
    gate = None

# 应用 gate
scaled_text = (effective_text_weight * gate) * text_raw
```

---

## 🎯 GATE 的组合效果

### 完整的 Gate 公式

```python
# 1. 全局 gate
alpha = sigmoid(text_gate_param)  # 可学习，范围 [0,1]

# 2. 对齐和温度缩放
align_scale = 1.0 + alignment_weight
temp_scale = 0.07 / temperature

# 3. 有效文本权重
effective_text_weight = alpha * text_weight * align_scale * temp_scale

# 4. Per-item gate (如果启用)
if text_item_gate_all is not None:
    gate = text_item_gate_all[item_ids]  # 每个 item 的 gate [0 或 1]
    final_weight = effective_text_weight * gate
else:
    final_weight = effective_text_weight

# 5. 应用到文本特征
scaled_text = final_weight * text_features
```

### 层级结构

```
Text Features
  ↓
× alpha (全局可学习 gate, 0.5 → 学习到的值)
  ↓
× text_weight (固定权重, 0.8)
  ↓
× align_scale (对齐缩放, ~1.05)
  ↓
× temp_scale (温度缩放, ~1.0)
  ↓
× gate (per-item gate, 0 或 1)
  ↓
Scaled Text Features
```

---

## 📋 配置文件中的 GATE 参数

在 `sasrec_align_qwen3.yaml` 中：

```yaml
# Line 84-86
text_gate_init: 0.5          # 全局 gate 初始值
text_gate_reg_l2: 0.05       # L2 正则化系数
text_gate_reg_entropy: 0.0   # 熵正则化系数（默认不用）

# Line 75-76
text_weight: 0.8             # 文本特征基础权重
text_tail_threshold: 5       # Per-item gate 阈值（交互次数）
```

---

## 🔬 GATE 的作用

### 1. 全局 Gate (`text_gate_param`)

**作用**: 自动学习文本特征的全局重要性
- 如果文本特征有用 → gate 学习到较大值（接近 1.0）
- 如果文本特征无用 → gate 学习到较小值（接近 0.0）

**训练中的正则化**:
```python
# Line 967-974
if self.text_gate_reg_l2 > 0.0:
    loss = loss + self.text_gate_reg_l2 * (alpha ** 2)  # 惩罚过大的 gate
```

### 2. Per-Item Gate (`text_item_gate_all`)

**作用**: 区分冷启动 item 和热门 item
- **冷启动 item** (交互 ≤ 5次): gate = 1.0，**充分利用文本特征**
- **热门 item** (交互 > 5次): gate = 0.0，**依赖 ID embedding**

**优势**:
- 冷启动 item 缺少交互信号，文本特征更重要
- 热门 item 已有充足交互信号，ID embedding 更准确

---

## 🆚 与 Multi-View 的 GATE 对比

| 特性 | SASRecAlign (TF-IDF+LLM) | SASRecAlignMultiView |
|------|--------------------------|---------------------|
| **全局 Gate** | ✅ `text_gate_param` | ✅ `text_gate_param` |
| **Per-Item Gate** | ✅ `text_item_gate_all` | ✅ `text_item_gate_all` |
| **Per-View Gate** | ❌ 无 (单个 LLM 特征) | ✅ `text_view_gate_params` (4个) |

**Multi-View 的额外 Gate**:

```python
# Line 617-619
self.text_view_gate_params = nn.Parameter(
    torch.full((num_views,), float(self.text_gate_init), dtype=torch.float32)
)
```

- 4 个可学习参数，每个视图一个
- 自动学习不同视图的重要性（Identity, Function, Audience, Category）

---

## ✅ 总结

### SASRecAlign (TF-IDF+LLM) 已有的 GATE

1. ✅ **全局可学习 Gate**: `text_gate_param`
   - 初始值: 0.5
   - 范围: [0, 1]
   - L2 正则: 0.05

2. ✅ **Per-Item Gate**: `text_item_gate_all`
   - 基于交互次数阈值（5次）
   - 冷启动 item 启用，热门 item 禁用

### 与之前修改的关系

在公平对比配置中，我们**新增了 SENet**，但 **GATE 本来就有**：

```yaml
# sasrec_align_qwen3.yaml (修改后)

# ✅ 已有的 GATE 配置（无需修改）
text_gate_init: 0.5
text_gate_reg_l2: 0.05
text_gate_reg_entropy: 0.0
text_weight: 0.8
text_tail_threshold: 5

# ⭐ 新增的 SENet 配置
text_use_senet: true    # 新增
num_text_views: 1       # 新增
```

### 结论

✅ **SASRecAlign 模型已经有完整的 GATE 机制**  
✅ **双层设计**: 全局 gate + per-item gate  
✅ **配置完善**: 初始值、正则化都可配置  
✅ **与 Multi-View 相比**: 缺少 per-view gates，但有相同的全局和 per-item gates

**因此，在公平对比配置中，两个模型的 GATE 机制是对齐的**（都有全局 + per-item，Multi-View 额外有 per-view gates，这是方法设计差异）。

---

**验证**: 查看训练日志中的 `text_gate_alpha` 值，应该会看到类似：
```
SASRecAlign: first-step text_gate_alpha=0.500000 (use_llm=True, use_cross=True, ...)
```

这证明 GATE 已经在工作！

