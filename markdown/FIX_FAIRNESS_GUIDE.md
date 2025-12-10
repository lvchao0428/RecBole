# 修复公平性问题 - 操作指南

## 问题

当前 Multi-View Split 模型使用 **768 维**融合，而 TF-IDF+LLM 使用 **512 维**融合，导致对比不公平。

## 解决方案

将 Multi-View 模型的融合维度从 768 降为 512，与 TF-IDF+LLM 对齐。

---

## 📝 修改步骤

### 步骤 1: 备份原始文件

```bash
cd /home/charlie/project/RecBole

# 备份模型文件
cp recbole/model/sequential_recommender/sasrecalignmultiview.py \
   recbole/model/sequential_recommender/sasrecalignmultiview.py.backup

# 备份配置文件
cp sasrec_align_multi_view.yaml sasrec_align_multi_view.yaml.backup
```

### 步骤 2: 修改模型代码

编辑文件：`recbole/model/sequential_recommender/sasrecalignmultiview.py`

#### 修改位置 1: Projection 层输出维度（约第 102 行）

**查找**:
```python
# Project to hidden_size * 2 (512) to align with dual-path model's fusion dimension
self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size * 2)
```

**替换为**:
```python
# Project to hidden_size (256) for fair comparison (fusion_dim = 512)
self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size)
```

#### 修改位置 2: Fusion 输入维度（约第 111 行）

**查找**:
```python
fusion_input_dim = self.hidden_size + self.hidden_size * 2  # 256 + 512 = 768
```

**替换为**:
```python
fusion_input_dim = self.hidden_size + self.hidden_size  # 256 + 256 = 512
```

#### 修改位置 3: 更新注释（约第 121 行）

**查找**:
```python
self.item_fusion_predictor = nn.Linear(
    fusion_input_dim + self.hidden_size,  # 768 + 256 = 1024
    self.hidden_size
)
```

**替换为**:
```python
self.item_fusion_predictor = nn.Linear(
    fusion_input_dim + self.hidden_size,  # 512 + 256 = 768
    self.hidden_size
)
```

#### 修改位置 4: Docstring 更新（约第 257-264 行）

**查找**:
```python
# Step 4: Project to hidden_size * 2 (512) to align with dual-path fusion dimension
# Input: [B, 1280] if base exists, else [B, 1024]
# Output: [B, 512]
text_proj = self.multiview_concat_proj(text_concat)
```

**替换为**:
```python
# Step 4: Project to hidden_size (256) for fair comparison
# Input: [B, 1280] if base exists, else [B, 1024]
# Output: [B, 256]
text_proj = self.multiview_concat_proj(text_concat)
```

#### 修改位置 5: Fusion 说明（约第 280-281 行）

**查找**:
```python
Args:
    item_emb: Item embeddings [B, hidden_size=256]
    text_raw: Projected multi-view (+ base) text features [B, hidden_size*2=512]
    item_ids: Item IDs [B]
```

**替换为**:
```python
Args:
    item_emb: Item embeddings [B, hidden_size=256]
    text_raw: Projected multi-view (+ base) text features [B, hidden_size=256]
    item_ids: Item IDs [B]
```

### 步骤 3: 更新配置注释

编辑文件：`sasrec_align_multi_view.yaml`

**查找**（约第 44-46 行）:
```yaml
# Text feature sources (TF-IDF base + 4-view Qwen3 split embeddings)
# Architecture: base[256] + multi-view[4×64→1024→512] = fusion[768]
# Aligned with dual-path model's fusion dimension
```

**替换为**:
```yaml
# Text feature sources (TF-IDF base + 4-view Qwen3 split embeddings)
# Architecture: base[256] + multi-view[4×64→256] = fusion[512]
# Fair comparison: same fusion dimension as TF-IDF+LLM baseline
```

### 步骤 4: 验证修改

```bash
# 1. 检查语法错误
python -c "from recbole.model.sequential_recommender.sasrecalignmultiview import SASRecAlignMultiView; print('✅ Import OK')"

# 2. 运行一个小测试
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml" \
  --phase_a_epochs 1 \
  --only_phase_a \
  --seed 2025

# 如果没报错，说明修改成功
```

---

## 🧪 完整测试流程

### 1. Baseline（原始不公平配置）

```bash
# TF-IDF+LLM
bash two_phase_run_tfidf_llm.sh
# 记录: Recall@10 = _____ (A)

# Multi-View (768 fusion)
bash two_phase_run_multiview_split.sh
# 记录: Recall@10 = _____ (B)

# 计算不公平优势
# Unfair Gain = (B - A) / A * 100%
```

### 2. 应用修复

```bash
# 应用上述所有修改
vim recbole/model/sequential_recommender/sasrecalignmultiview.py
vim sasrec_align_multi_view.yaml
```

### 3. 重新测试（公平配置）

```bash
# Multi-View (512 fusion - Fair)
bash two_phase_run_multiview_split.sh
# 记录: Recall@10 = _____ (C)

# 计算公平优势
# Fair Gain = (C - A) / A * 100%

# 计算架构带来的额外增益
# Architecture Gain = (B - C) / A * 100%
```

---

## 📊 结果对比模板

创建文件 `fairness_comparison_results.txt`:

```
===== Multi-View vs TF-IDF+LLM 公平性对比 =====

日期: YYYY-MM-DD
数据集: Amazon_Beauty

[1] TF-IDF+LLM Baseline (512 fusion)
  - Recall@10:  _____
  - NDCG@10:    _____
  - MRR@10:     _____

[2] Multi-View 原始配置 (768 fusion) - ❌ 不公平
  - Recall@10:  _____ (+___%)
  - NDCG@10:    _____ (+___%)
  - MRR@10:     _____ (+___%)
  - 融合维度: 768 (比baseline大50%)
  - 额外模块: SENet + Per-View Gates

[3] Multi-View 公平配置 (512 fusion) - ✅ 公平
  - Recall@10:  _____ (+___%)
  - NDCG@10:    _____ (+___%)
  - MRR@10:     _____ (+___%)
  - 融合维度: 512 (与baseline相同)
  - 额外模块: SENet + Per-View Gates

[4] 性能分析
  - 总提升 (不公平): [2] vs [1] = _____%
  - 公平提升:        [3] vs [1] = _____%
  - 架构优势:        [2] vs [3] = _____%
  
  结论:
  - 多视图方法本身贡献: _____%
  - 更大融合维度贡献:    _____%
  - SENet+Gates 贡献:    _____ % (需额外消融实验)

[5] 下一步
  [ ] 消融实验: 移除 SENet
  [ ] 消融实验: 移除 Per-View Gates
  [ ] 消融实验: 固定均匀视图权重
```

---

## 🔧 快速回滚

如果修改后出现问题：

```bash
# 恢复原始文件
cp recbole/model/sequential_recommender/sasrecalignmultiview.py.backup \
   recbole/model/sequential_recommender/sasrecalignmultiview.py

cp sasrec_align_multi_view.yaml.backup \
   sasrec_align_multi_view.yaml

# 验证回滚
python -c "from recbole.model.sequential_recommender.sasrecalignmultiview import SASRecAlignMultiView; print('✅ Rollback OK')"
```

---

## 📝 修改前后对比

### 关键维度变化

| 阶段 | 修改前 | 修改后 | 说明 |
|------|-------|--------|------|
| **Multi-view concat** | [B, 1024] | [B, 1024] | 不变 |
| **+ Base (TF-IDF)** | [B, 1280] | [B, 1280] | 不变 |
| **Projection output** | [B, **512**] | [B, **256**] | ⭐ 减半 |
| **Fusion input** | [B, **768**] | [B, **512**] | ⭐ 对齐 |
| **Cross Network size** | **768** | **512** | ⭐ 对齐 |
| **Predictor input** | [B, 1024] | [B, 768] | 相应调整 |

### 参数量变化

```
修改前:
  - multiview_concat_proj: 1280 × 512 = 655K
  - item_fusion_cross:     768 × 768 × 2 = 1.18M
  - item_fusion_deep:      768 × 256 = 197K
  - item_fusion_predictor: 1024 × 256 = 262K
  Total: ~2.3M

修改后:
  - multiview_concat_proj: 1280 × 256 = 328K  (-50%)
  - item_fusion_cross:     512 × 512 × 2 = 524K  (-56%)
  - item_fusion_deep:      512 × 256 = 131K  (-34%)
  - item_fusion_predictor: 768 × 256 = 197K   (-25%)
  Total: ~1.2M  (-48%)
```

**减少约 48% 的参数量** - 这使得对比更公平！

---

## ⚠️ 常见问题

### Q1: 修改后性能显著下降怎么办？

**A**: 这是预期的！说明之前的提升部分来自更大的模型。
- 如果仍有 1-2% 提升：说明多视图方法本身有价值
- 如果没有提升：说明之前的提升主要来自模型容量

### Q2: 修改后还需要调整超参数吗？

**A**: 建议保持超参数不变（Grid Search 会自动搜索最佳参数）
- 但可以考虑增加 `lr_text_head`（因为参数变少了）
- Phase B 的 `text_weight` 可能需要微调

### Q3: SENet 和 Gates 还保留吗？

**A**: 是的，这次修改只统一融合维度。
- SENet 和 Gates 是多视图方法的一部分
- 它们的贡献需要额外的消融实验来评估
- 如果想完全公平，需要单独移除它们

### Q4: 修改是否会影响其他实验？

**A**: 只影响使用 `SASRecAlignMultiView` 模型的实验
- `SASRec_Align` (TF-IDF+LLM) 不受影响
- `SASRecAlign` (其他配置) 不受影响
- 只有 Multi-View Split 相关实验受影响

---

## ✅ 检查清单

修改完成后，请确认：

- [ ] 备份了原始文件
- [ ] 修改了所有 5 个位置
- [ ] 更新了配置文件注释
- [ ] 通过了语法检查
- [ ] 运行了小规模测试
- [ ] 记录了修改前的性能
- [ ] 准备了结果对比表格
- [ ] 理解了预期的性能变化

---

**修改完成后，运行**:
```bash
bash two_phase_run_multiview_split.sh
```

**对比结果，分析真实的多视图方法价值！**

---

**帮助**: 如有问题，对比备份文件查看差异
```bash
diff -u recbole/model/sequential_recommender/sasrecalignmultiview.py.backup \
        recbole/model/sequential_recommender/sasrecalignmultiview.py
```

