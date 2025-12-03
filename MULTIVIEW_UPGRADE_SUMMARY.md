# Multi-View Split 架构升级总结

## 📅 更新日期
2025-12-02

## 🎯 升级目标
将原有的简单加权融合架构升级为**每个视角独立对齐**的多视角学习架构，解决之前效果不佳的问题。

---

## 🔧 主要改动

### 1. 生成 4-View Qwen3 Embeddings

#### 新增脚本
- **文件**: `tools/gen_multiview_4views_beauty.sh`
- **功能**: 使用 `build_item_text_emb_qwen3_hf.py` 生成4个视角的分离embeddings

#### 运行命令
```bash
cd /home/charlie/project/RecBole
bash tools/gen_multiview_4views_beauty.sh
```

#### 生成的文件
```
dataset/Amazon_Beauty/
├── item_text_emb_qwen3_4views.npy                    # Concat版本（backup）
└── item_text_emb_qwen3_4views_split/
    ├── views.json                                     # 元数据
    ├── view_0.npy  # View 1: Identity/Title
    ├── view_1.npy  # View 2: Function/Features  
    ├── view_2.npy  # View 3: Target Audience
    └── view_3.npy  # View 4: Category/Context
```

#### 4个视角的Prompt定义
1. **Identity/Title**: "Identify the item: [TITLE] {text}"
2. **Function/Features**: "What are the main functions and features of [TITLE] {text}?"
3. **Target Audience**: "Who is the target audience or user group for [TITLE] {text}?"
4. **Category/Context**: "Categorize the item [TITLE] {text} and describe its context."

---

### 2. 模型架构重构

#### 文件修改
- **文件**: `recbole/model/sequential_recommender/sasrecalignmultiview.py`

#### 核心改动

##### A. 新增可学习参数
```python
# Per-view gate parameters (for weighted fusion)
self.text_view_gate_params = nn.Parameter(torch.zeros(num_views))

# Per-view alignment weights (learnable)
self.text_view_align_weights = nn.Parameter(torch.ones(num_views))

# Multi-view concat projection
self.multiview_concat_proj = nn.Linear(num_views * hidden_size, hidden_size)
```

##### B. 新的融合流程
```
原架构（错误）:
多个view → SENet → 加权求和pooling → residual融合 → item_emb

新架构（正确）:
多个view → SENet增强 → Gate加权 → Concat拼接 → Projection投影 → Cross Network融合
  ↓
每个view独立计算对齐损失（可学习权重）
```

##### C. Per-View Alignment Loss
```python
def calculate_loss(self, interaction):
    loss = super().calculate_loss(interaction)
    
    # 计算每个view和ID embedding的对齐损失
    for idx in range(num_views):
        view_feat = view_stack[:, idx, :]
        align_loss_i = self._info_nce_align(id_emb, view_feat)
        per_view_losses.append(align_loss_i)
    
    # 使用可学习权重加权
    align_weights = F.softmax(self.text_view_align_weights, dim=0)
    total_align_loss = sum(w_i * loss_i for w_i, loss_i)
    
    return loss + alignment_weight * total_align_loss
```

---

### 3. 配置文件更新

#### 文件修改
- **文件**: `sasrec_align_multi_view.yaml`

#### 关键配置变更

| 配置项 | 旧值 | 新值 | 说明 |
|--------|------|------|------|
| `item_text_emb_split_dir` | `item_text_emb_amplified_views` | `item_text_emb_qwen3_4views_split` | 指向新的4-view目录 |
| `num_text_views` | 5 | 4 | 减少到4个视角 |
| `use_cross` | false | true | 启用Cross Network融合 |
| `text_cross_layer_num` | - | 2 | 添加Cross层配置 |
| `cosine_score` | false | true | 启用余弦相似度打分 |
| `fuse_text_feature` | - | true | 显式启用文本融合 |

#### 对齐 `sasrec_align_qwen3.yaml` 的配置
- ✅ Cross Network融合
- ✅ Cosine scoring
- ✅ 相同的正则化参数
- ✅ 相同的dropout配置

---

### 4. 训练脚本更新

#### 文件修改
- **文件**: `two_phase_run_multiview_split.sh`

#### 主要调整
```bash
# 更新注释，说明新架构
# 调整超参数网格，对齐tfidf_llm实验
--align_grid "0.01,0.02,0.05,0.08"  # 扩展搜索范围
--tau_grid "0.03,0.05,0.07"         # 对齐tfidf_llm

# 更新variant标记
--variant_features "sasrec,multiview,4views,per_view_align,qwen3"

# 更新checkpoint目录
--checkpoint_dir ./saved/phase_runs_multiview_4views
```

---

## 📊 架构对比

### 旧架构的问题
1. ❌ 没有使用LLM embedding（只用TF-IDF）
2. ❌ 多视角简单加权求和，损失语义信息
3. ❌ 单一对齐信号，无法充分利用多视角
4. ❌ 缺少Cross Network，融合能力弱

### 新架构的优势
1. ✅ **Per-view alignment**: 每个视角独立优化与ID的对齐关系
2. ✅ **Learnable weights**: 可学习的视角重要性权重
3. ✅ **Concat fusion**: 保留所有视角的语义信息
4. ✅ **Cross Network**: 强大的非线性融合能力
5. ✅ **Architecture alignment**: 与成功的tfidf+llm实验完全对齐

---

## 🚀 使用流程

### Step 1: 生成Multi-View Embeddings
```bash
cd /home/charlie/project/RecBole

# 如果没有item_index_mapping.csv，先生成
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_index_mapping.csv

# 生成4-view embeddings
bash tools/gen_multiview_4views_beauty.sh
```

### Step 2: 验证Embeddings
```bash
# 检查生成的文件
ls -lh dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split/

# 查看元数据
cat dataset/Amazon_Beauty/item_text_emb_qwen3_4views_split/views.json
```

### Step 3: 运行训练
```bash
# 启动两阶段训练
bash two_phase_run_multiview_split.sh
```

### Step 4: 监控训练
```bash
# 查看日志
tail -f run_metrics/watchdog_multiview_4views.log

# 检查checkpoint
ls -lh saved/phase_runs_multiview_4views/
```

---

## 🔍 关键设计决策

### 1. Temperature共享
- **决策**: 4个view共享一个temperature参数
- **原因**: 减少超参数数量，简化调优

### 2. 对齐权重可学习
- **决策**: 使用`nn.Parameter`让模型学习每个view的重要性
- **实现**: `F.softmax(self.text_view_align_weights)`确保权重归一化

### 3. Gate机制保留
- **决策**: 保留per-view gate参数
- **作用**: 在融合前控制每个view的贡献度

### 4. 融合顺序确定
- **决策**: SENet → Gate → Concat → Projection → Cross Network
- **原因**: 
  - SENet提升特征质量
  - Gate控制重要性
  - Concat保留完整信息
  - Projection降维
  - Cross Network强融合

---

## 📈 预期效果

### 对比基线
根据之前的实验结果：

| 配置 | Recall@10 | NDCG@10 | MRR@10 |
|------|-----------|---------|--------|
| sasrec baseline | 0.0562 | 0.0292 | 0.0207 |
| tfidf only | 0.0615 | 0.0309 | 0.0214 |
| tfidf + llm | 0.0648 | 0.0390 | 0.0311 |
| multiview concat | 0.0628 | 0.0325 | 0.0231 |
| **旧 multiview split** | **0.0500** | **0.0270** | **0.0198** |

### 新架构预期
- 应该**超越 tfidf only**（+9-10%）
- 有望**接近或超越 multiview concat**（+11-12%）
- 可能**接近 tfidf + llm**（理想情况）

### 关键指标
- **MRR提升**: Per-view alignment应显著提升排序质量
- **NDCG提升**: Cross Network融合应改善整体排序
- **Recall稳定**: Gate机制保证召回的鲁棒性

---

## 🐛 潜在问题与解决

### 问题1: 内存占用增加
- **原因**: 4个view + SENet + Cross Network
- **解决**: 已配置 `fusion_chunk_size: 65536`

### 问题2: 训练速度变慢
- **原因**: Per-view alignment loss计算
- **解决**: 只在training时计算，并detach不需要梯度的部分

### 问题3: 过拟合风险
- **原因**: 参数量增加
- **解决**: 
  - Cross dropout: 0.5
  - Text gate L2正则: 0.05
  - Label smoothing: 0.1

---

## 📝 代码维护建议

### 关键文件
1. `recbole/model/sequential_recommender/sasrecalignmultiview.py` - 核心模型
2. `sasrec_align_multi_view.yaml` - 配置文件
3. `two_phase_run_multiview_split.sh` - 训练脚本
4. `tools/gen_multiview_4views_beauty.sh` - 数据生成

### 调试技巧
```python
# 在训练日志中查看per-view alignment信息
# 第一个batch会打印:
# - Per-view alignment weights: [w0, w1, w2, w3]
# - Per-view alignment losses: [L0, L1, L2, L3]
```

### 扩展到其他数据集
```bash
# 修改gen_multiview_4views_beauty.sh
# 替换所有 Amazon_Beauty 为目标数据集名称
# 例如: yelp, Amazon_Toys_and_Games, etc.
```

---

## ✅ 验证清单

部署前请确认：
- [ ] 4-view embeddings已生成（`views.json`存在）
- [ ] 配置文件正确指向新的split目录
- [ ] `use_cross: true` 已启用
- [ ] `num_text_views: 4` 已更新
- [ ] 训练脚本的超参数网格已更新

---

## 📞 问题反馈

如遇到问题，检查：
1. 生成的embeddings维度是否正确（每个view 128维）
2. `views.json`的`num_prompts`是否为4
3. 模型日志中是否有per-view alignment信息
4. Cross Network是否正确初始化

---

**升级完成！祝训练顺利！🚀**

