# Toys数据集部署指南

基于Beauty数据集的成功经验（**Recall@10: +12.28%, MRR@10: +50.24%**），快速部署到Toys_and_Games数据集。

---

## 📊 **Beauty数据集成功验证**

### 最新结果（2025-12-03）

| 配置 | Recall@10 | NDCG@10 | MRR@10 |
|------|-----------|---------|--------|
| Baseline | 0.0562 | 0.0292 | 0.0207 |
| **Split Multi-View + Gate** | **0.0631** (+12.28%) | **0.0386** (+32.19%) | **0.0311** (+50.24%) |

### 关键优势
✅ **综合最优**: 在Recall、NDCG、MRR上都接近或达到最佳  
✅ **MRR显著提升**: +50.24%，说明Top-1准确率大幅提升  
✅ **可解释性强**: Per-view权重可分析每个视角的贡献  

---

## 🚀 **快速部署（3步）**

### Step 1: 生成Multi-View Embeddings

```bash
cd /home/charlie/project/RecBole

# 如果没有item_index_mapping.csv，先生成
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Toys_and_Games/item_index_mapping.csv

# 生成4-view embeddings（约需30-60分钟）
bash tools/gen_multiview_4views_toys.sh
```

### Step 2: 验证数据文件

```bash
# 检查生成的文件
ls -lh dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split/

# 应该看到:
# - views.json (元数据)
# - view_0.npy, view_1.npy, view_2.npy, view_3.npy (4个视角)

# 查看元数据
cat dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split/views.json
```

### Step 3: 启动训练

```bash
# 启动两阶段训练
bash two_phase_run_multiview_split_toys.sh

# 监控训练（另开终端）
tail -f run_metrics/watchdog_multiview_4views_toys.log
```

---

## 📂 **文件清单**

### 数据生成
- `tools/gen_multiview_4views_toys.sh` - 生成4-view embeddings的脚本

### 配置文件
- `sasrec_align_multi_view_toys.yaml` - 模型配置（基于Beauty配置）

### 训练脚本
- `two_phase_run_multiview_split_toys.sh` - 两阶段训练脚本

### 输出目录
- `dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split/` - Multi-view embeddings
- `saved/phase_runs_multiview_4views_toys/` - 训练checkpoints
- `run_metrics/watchdog_multiview_4views_toys.log` - 训练监控日志

---

## 🔍 **关键配置说明**

### 与Beauty数据集的主要区别

| 配置项 | Beauty | Toys | 说明 |
|--------|--------|------|------|
| `dataset` | Amazon_Beauty | Amazon_Toys_and_Games | 数据集名称 |
| `ndcg_baseline` | 0.0272 | 0.025 | 根据Toys baseline调整 |
| 其他配置 | 完全相同 | 完全相同 | 复用成功经验 |

### 超参数网格（与Beauty一致）

```bash
--align_grid "0.01,0.02,0.05,0.08"  # 对齐权重搜索
--tau_grid "0.03,0.05,0.07"         # 温度参数搜索
```

---

## 📈 **预期效果**

基于Beauty数据集的经验，Toys数据集预期：

| 指标 | 预期提升 | 说明 |
|------|---------|------|
| **Recall@10** | +10-15% | 召回率稳定提升 |
| **NDCG@10** | +30-35% | 排序质量显著改善 |
| **MRR@10** | +45-55% | Top-1准确率大幅提升 |

**关键因素**：
- ✅ Per-view alignment: 每个视角独立优化
- ✅ Learnable weights: 自动学习视角重要性
- ✅ Cross Network: 强大的非线性融合

---

## 🐛 **常见问题**

### Q1: 生成embeddings时内存不足？
```bash
# 方案1: 减小batch_size
--batch_size 8  # 默认16

# 方案2: 启用更激进的内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
```

### Q2: 训练时GPU内存不足？
```yaml
# 修改 sasrec_align_multi_view_toys.yaml
fusion_chunk_size: 32768  # 默认65536，减半
train_batch_size: 256     # 默认512，减半
```

### Q3: 如何查看per-view权重？
```bash
# 查看训练日志的第一个batch输出
grep "per-view alignment enabled" run_metrics/watchdog_multiview_4views_toys.log

# 输出示例:
# weights=[w0=0.28, w1=0.24, w2=0.25, w3=0.23]
# → View 0 (Identity)最重要
```

---

## 🔬 **实验记录模板**

```
实验日期: ____
数据集: Amazon_Toys_and_Games
配置: split multi-view + gate

Phase A 最佳超参:
- alignment_weight: ____
- temperature: ____

Phase B 结果:
- Recall@10: ____ (相对baseline: +___%)
- NDCG@10: ____ (相对baseline: +___%)
- MRR@10: ____ (相对baseline: +___%)

与Beauty对比:
- Beauty Recall@10提升: +12.28%
- Toys Recall@10提升: +____%
- 差异分析: ________________
```

---

## 📞 **检查清单**

部署前确认：
- [ ] item_index_mapping.csv 已生成
- [ ] 4-view embeddings 已生成（views.json存在）
- [ ] 配置文件指向正确的split目录
- [ ] GPU内存足够（建议24GB+）
- [ ] 磁盘空间足够（约10-20GB for embeddings + checkpoints）

训练中监控：
- [ ] 第一个batch有per-view alignment信息
- [ ] Phase A网格搜索正常运行
- [ ] Phase B从最佳超参继续训练
- [ ] Watchdog未报告内存警告

---

## 📊 **预期训练时间**

基于Beauty数据集经验（单GPU A100/V100）：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 生成embeddings | 30-60分钟 | 取决于数据集大小 |
| Phase A (burnin) | 2-3小时 | 10 epochs |
| Phase A (grid) | 6-10小时 | 20 epochs × 12 configs |
| Phase B | 8-12小时 | 40 epochs |
| **总计** | **~20-25小时** | 包含数据生成 |

---

**祝实验顺利！期待Toys数据集也能取得和Beauty相当的提升！** 🚀

