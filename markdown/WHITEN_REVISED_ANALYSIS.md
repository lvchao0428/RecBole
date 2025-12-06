# 白化特征效果变差的修正分析

## ✅ 排除的原因：L2归一化

经确认，模型在加载text embedding时会自动L2归一化（如果`normalize_text: true`）：

```python
# sasrec_align.py 第154-163行
if self.normalize_text:
    with torch.no_grad():
        if emb_base is not None:
            norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
            emb_base = emb_base / norms.clamp_min(1e-8)
```

所以即使白化后的embedding norm=34，模型加载后也会变成norm≈1，**L2归一化不是问题**。

---

## 🎯 真正的问题：白化改变了特征的几何结构

### 实验结果模式分析：

| 指标 | Before | After | 变化 | 含义 |
|------|--------|-------|------|------|
| Recall@10 | 0.0677 | 0.0511 | **-24.5%** | 召回范围缩小 |
| MRR@10 | 0.0238 | 0.0285 | **+19.7%** | Top-1更准确 |
| NDCG@10 | 0.0342 | 0.0338 | -1.2% | 综合持平 |

**关键观察**：
- ✅ MRR大幅提升 → 模型对top-1位置的预测更自信、更准确
- ❌ Recall大幅下降 → 模型漏掉了很多真正相关的物品
- ⚖️ NDCG基本持平 → 提升和损失相互抵消

**这是典型的"过度自信"模式**：模型变得保守，只推荐少数高置信度的物品。

---

## 🔬 根本原因分析

### 原因1: 白化改变了特征的信息结构（主要原因）

**原始TF-IDF特征**：
- 有明显的主成分（高方差维度）和次成分（低方差维度）
- 主成分捕捉主要语义信号
- 次成分可能是噪声或细微差别

**白化后的特征**：
- 所有维度的方差相等（Cov = I）
- 主成分和次成分被"拉平"了
- 所有维度权重相等

**影响**：
```
原始：主要信号（大方差）主导相似度计算
白化：所有维度平等 → 细微差别被放大 → 相似度分布变窄
```

**具体表现**：
1. 正样本的相似度可能更集中（方差小）→ MRR高
2. 边界样本（中等相关）的相似度被压低 → Recall低
3. 模型倾向于"保守"推荐，只推高置信度的

---

### 原因2: Phase-A过拟合（次要但关键）

你提到"删除early jump phaseA"，这意味着：
- Phase-A完整跑8 epochs，没有提前终止
- Text projection head可能过度优化对齐任务
- 模型学到了过于"刚性"的text-ID对应关系

**对比**：
- ✅ Early stop: 达到阈值就停 → 保留一定灵活性
- ❌ No early stop: 跑满epochs → 过度拟合 → 泛化能力下降

---

### 原因3: 白化后的相似度分布改变

即使都做了L2归一化，白化后的cosine相似度分布可能不同：

**数学分析**：
```
原始TF-IDF (L2-normed):
  sim(i, j) = x_i^T x_j  (保留原始特征相关性)

白化后 (L2-normed):
  x_white = (x - μ) @ W
  sim_white(i, j) = x_white_i^T x_white_j
  
  由于W改变了特征空间的基，相似度分布会改变
```

**可能的影响**：
- 相似度的均值和方差改变
- Temperature=0.07可能不再适合新的分布
- 需要重新调整temperature来match新的相似度范围

---

## 🔧 修正后的解决方案

### 方案1: 恢复Phase-A Early Stopping（推荐优先尝试 ★★★★★）

**原理**：防止text projection head过拟合

**修改 `two_tfidf_stage.sh`**：

```bash
python scripts/exp_align_two_stage.py \
  --dataset Amazon_Beauty \
  --mode base \
  --stage1_epochs 6 \              # 8 → 6 (适度减少)
  --stage2_epochs 50 \
  --stage2_lr 3e-5 \
  --temperatures 0.05 0.07 0.1 \   # 扩大搜索范围
  --weights 0.05 0.1 \
  --exclude_topk 0 \
  --base_emb dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --ndcg_baseline 0.0342 \         # 使用NDCG作为gate (更平衡的指标)
  --ndcg_gain_threshold 0.015      # 要求提升1.5%即可
```

**关键改进**：
1. ✅ 减少Phase-A epochs: 8 → 6
2. ✅ 添加early stopping with NDCG gate
3. ✅ Temperature grid包含原始值0.07
4. ✅ 使用NDCG而非MRR作为gate (更平衡Recall和Ranking)

---

### 方案2: 调整Alignment Weight（快速测试 ★★★★）

**原理**：减少对text signal的依赖，恢复多样性

**测试配置**：

```yaml
# sasrec_align_base.yaml 修改
alignment_weight: 0.03   # 0.1 → 0.03 (减小50%)
temperature: 0.07        # 保持不变
normalize_text: true     # 保持L2归一化
```

**预期效果**：
- 减少对text的依赖 → 模型不会过度"保守"
- 恢复一定的探索性 → Recall提升
- 可能MRR略微下降，但NDCG和Recall应该提升

---

### 方案3: 调整Cross层的Dropout（防止过拟合 ★★★）

**原理**：防止text fusion模块过拟合

```yaml
# sasrec_align_base.yaml 修改
cross_dropout_prob: 0.3   # 0.0 → 0.3
hidden_dropout_prob: 0.5  # 保持不变
```

**预期效果**：
- Cross层加入dropout → 防止过度依赖text特定模式
- 提升泛化能力 → Recall恢复

---

### 方案4: 混合策略（最稳妥 ★★★★★）

综合以上方案，创建新配置：

```yaml
---
# sasrec_align_base_whiten_v2.yaml
model: SASRecAlign
hidden_size: 256

# Dataset & Training (保持不变)
dataset: Amazon_Beauty
epochs: 200
train_batch_size: 512
learning_rate: 0.0001
eval_step: 5
stopping_step: 20

# Text alignment settings (关键调整)
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy
use_llm: false
use_align: true
use_cross: true
text_cross_layer_num: 3

# 关键参数调整
alignment_weight: 0.05         # 0.1 → 0.05 (减少text依赖)
temperature: 0.07              # 保持不变 (先试试原来的)
normalize_text: true           # 确保L2归一化

# 防止过拟合的策略
cross_dropout_prob: 0.3        # 增加dropout
hidden_dropout_prob: 0.5
token_dropout_prob: 0.15       # 0.1 → 0.15 (增加序列dropout)

# Text fusion
text_weight: 0.3               # 0.4 → 0.3 (减少text权重)
text_tail_threshold: 5
fuse_text_feature: true

# 其他参数保持不变
detach_text_emb: true
cosine_score: true
cosine_scale: 10.0
weight_decay: 1e-5
label_smoothing: 0.1
```

**训练脚本**：

```bash
#!/bin/bash
# two_phase_whiten_revised.sh

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_whiten_v2.yaml" \
  \
  --phase_a_epochs 5 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "NDCG@10" \
  \
  --phase_b_epochs 40 \
  \
  --lr_text_head 0.001 \
  --lr_dnn_cross 0.0005 \
  --backbone_lr_scale 0.1 \
  \
  --phase_a_grid \
  --align_grid "0.03,0.05,0.08" \
  --tau_grid "0.05,0.07,0.1" \
  \
  --ndcg_baseline 0.0342 \
  --ndcg_gain_threshold 0.015 \
  --phase_a_auto_to_b \
  \
  --save \
  --checkpoint_dir "./saved/whiten_revised"
```

---

## 🎯 推荐执行顺序

### Step 1: 验证normalize_text确实生效

检查你之前实验用的配置文件，确认：

```bash
grep "normalize_text" <你的配置文件.yaml>
```

应该看到 `normalize_text: true`

### Step 2: 快速测试 - 调整alignment_weight

```bash
python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_base.yaml \
  --config_dict "alignment_weight=0.03,cross_dropout_prob=0.3"
```

**预期**：如果Recall恢复，说明确实是过度依赖text

### Step 3: 完整两阶段训练（推荐配置）

```bash
bash two_phase_whiten_revised.sh
```

**关键改进**：
1. Phase-A只跑5 epochs (防止过拟合)
2. 使用NDCG@10作为gate (平衡指标)
3. Alignment weight grid: 0.03/0.05/0.08 (更保守)
4. Temperature grid: 0.05/0.07/0.1 (包含原值)
5. 增加dropout防止过拟合

### Step 4: 对比分析

对比以下配置的结果：
1. 原始TF-IDF (baseline)
2. Whitened + alignment_weight=0.1 (当前差的结果)
3. Whitened + alignment_weight=0.05 (新配置)
4. Whitened + Phase-A 5 epochs (新训练策略)

---

## 📊 预期结果

使用修正方案后：

| 指标 | Baseline | Whiten(旧) | Whiten(新) | 目标 |
|------|----------|------------|------------|------|
| Recall@10 | 0.0677 | 0.0511 | **≥0.065** | 恢复95%+ |
| MRR@10 | 0.0238 | 0.0285 | **≥0.027** | 保持提升 |
| NDCG@10 | 0.0342 | 0.0338 | **≥0.036** | 综合提升 |

**理想情况**：
- Recall恢复到接近baseline
- MRR保持一定提升（白化的好处）
- NDCG整体提升（平衡性更好）

---

## 🔬 进一步诊断建议

### 1. 分析相似度分布

添加代码在训练时记录：
```python
# 在calculate_loss中
pos_sim = F.cosine_similarity(seq_output, pos_items_emb, dim=-1)
self.logger.info(f"Pos sim: mean={pos_sim.mean():.4f}, std={pos_sim.std():.4f}")
```

对比原始TF-IDF vs Whitened的相似度分布差异

### 2. 可视化embedding空间

```python
from sklearn.manifold import TSNE

# 对比原始 vs 白化后的embedding分布
tsne = TSNE(n_components=2)
emb_2d_orig = tsne.fit_transform(tfidf_emb[:1000])
emb_2d_white = tsne.fit_transform(whitened_emb[:1000])

# 绘制散点图，观察聚类结构
```

### 3. 检查Phase-A的训练曲线

查看Phase-A的validation NDCG曲线：
- 如果前3-4个epoch就plateau → 应该early stop
- 如果一直上升到8 epoch → 可能需要更多epochs
- 如果先升后降 → 明显过拟合

---

## 📌 关键结论

1. **L2归一化不是问题** - 模型会自动处理
2. **白化改变了特征几何** - 主成分被拉平，影响相似度分布
3. **模型变得过度保守** - MRR↑但Recall↓是典型的过拟合表现
4. **Phase-A可能过拟合** - 删除early stop让text head学得太"死"
5. **需要降低text依赖** - 减小alignment_weight和text_weight

**最关键的修改**：
- ✅ Phase-A epochs: 8 → 5
- ✅ Alignment weight: 0.1 → 0.05
- ✅ Cross dropout: 0.0 → 0.3
- ✅ 使用NDCG@10作为early stop指标（更平衡）

