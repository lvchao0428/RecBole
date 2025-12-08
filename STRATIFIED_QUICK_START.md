# 分档评估 - 快速开始指南

## 🎯 一句话总结

已为所有 8 个训练脚本添加**按交互次数分档评估**（new/few/frequent），可直接运行查看细粒度性能。

---

## 📊 分档定义

```
new:      [1, 3)   交互次数 1-2 次（新 item，冷启动）
few:      [3, 10)  交互次数 3-9 次（少量交互）
frequent: [10, +∞) 交互次数 ≥10 次（频繁交互，热门）
```

---

## 🚀 快速使用

### Beauty 数据集

```bash
cd /home/charlie/project/RecBole

# Baseline (70 epochs, pure SASRec)
bash run70epBase_stratified.sh

# TF-IDF
bash two_phase_run_tfidf_stratified.sh

# TF-IDF + LLM (Qwen3)
bash two_phase_run_tfidf_llm_stratified.sh

# Multi-View Split
bash two_phase_run_multiview_split_stratified.sh
```

### Toys 数据集

```bash
# Baseline
bash run70epBase_toys_stratified.sh

# TF-IDF
bash two_phase_run_tfidf_toys_stratified.sh

# TF-IDF + LLM
bash two_phase_run_tfidf_llm_toys_stratified.sh

# Multi-View
bash two_phase_run_multiview_split_toys_stratified.sh
```

---

## 📈 新增指标（每个脚本都会输出）

### 标准指标（原有）
```
Recall@10, NDCG@10, MRR@10, Hit@10, Precision@10
```

### 分档指标（新增）⭐
```
# 分档 Recall
Recall_new@10        # 新 item 召回率
Recall_few@10        # 少量交互 item 召回率
Recall_frequent@10   # 频繁交互 item 召回率

# 分档 NDCG
NDCG_new@10
NDCG_few@10
NDCG_frequent@10

# 推荐覆盖率
Coverage_new@10      # Top-10 中新 item 的比例
Coverage_few@10      # Top-10 中少量交互 item 的比例
Coverage_frequent@10 # Top-10 中频繁交互 item 的比例
```

---

## 📁 文件清单（16个）

### 配置文件（8个）

**Beauty 数据集**:
1. `sasrec_baseline_70ep_stratified.yaml`
2. `sasrec_align_base_stratified.yaml`
3. `sasrec_align_qwen3_stratified.yaml`
4. `sasrec_align_multi_view_stratified.yaml`

**Toys 数据集**:
5. `sasrec_baseline_70ep_toys_stratified.yaml`
6. `sasrec_align_toys_base_stratified.yaml`
7. `sasrec_align_toys_qwen3_stratified.yaml`
8. `sasrec_align_multi_view_toys_stratified.yaml`

### 训练脚本（8个）

**Beauty 数据集**:
1. `run70epBase_stratified.sh`
2. `two_phase_run_tfidf_stratified.sh`
3. `two_phase_run_tfidf_llm_stratified.sh`
4. `two_phase_run_multiview_split_stratified.sh`

**Toys 数据集**:
5. `run70epBase_toys_stratified.sh`
6. `two_phase_run_tfidf_toys_stratified.sh`
7. `two_phase_run_tfidf_llm_toys_stratified.sh`
8. `two_phase_run_multiview_split_toys_stratified.sh`

---

## 🔍 查看结果

### 训练日志中的输出

```
==================== Test Result ====================
Recall@10: 0.0315
NDCG@10: 0.0189
MRR@10: 0.0412

# ⭐ 分档指标
Recall_new@10: 0.0523        # 新 item 性能
Recall_few@10: 0.0387
Recall_frequent@10: 0.0256

NDCG_new@10: 0.0298
NDCG_few@10: 0.0223
NDCG_frequent@10: 0.0167

Coverage_new@10: 0.1523      # 推荐多样性
Coverage_few@10: 0.3245
Coverage_frequent@10: 0.5232
=====================================================
```

### 关键观察点

1. **Recall_new vs Recall_frequent**
   - 如果 `Recall_new >> Recall_frequent` → 文本特征对冷启动非常有效
   - 如果 `Recall_new ≈ Recall_frequent` → 文本特征对所有 item 都有效

2. **Coverage_new**
   - 如果 `Coverage_new` 较高（>15%）→ 推荐多样性好
   - 如果 `Coverage_new` 很低（<5%）→ 推荐偏向热门

3. **Multi-View vs TF-IDF+LLM**
   - 比较各档的性能差异
   - 判断多视图方法在哪些场景下更有效

---

## 💡 实验建议

### 并行运行（如果有多 GPU）

**Beauty**:
```bash
CUDA_VISIBLE_DEVICES=0 bash run70epBase_stratified.sh &
CUDA_VISIBLE_DEVICES=1 bash two_phase_run_tfidf_stratified.sh &
CUDA_VISIBLE_DEVICES=2 bash two_phase_run_tfidf_llm_stratified.sh &
CUDA_VISIBLE_DEVICES=3 bash two_phase_run_multiview_split_stratified.sh &
wait
```

**Toys**:
```bash
CUDA_VISIBLE_DEVICES=0 bash run70epBase_toys_stratified.sh &
CUDA_VISIBLE_DEVICES=1 bash two_phase_run_tfidf_toys_stratified.sh &
CUDA_VISIBLE_DEVICES=2 bash two_phase_run_tfidf_llm_toys_stratified.sh &
CUDA_VISIBLE_DEVICES=3 bash two_phase_run_multiview_split_toys_stratified.sh &
wait
```

### 串行运行（单 GPU）

```bash
# 依次运行所有实验
for script in run70epBase_stratified.sh \
              two_phase_run_tfidf_stratified.sh \
              two_phase_run_tfidf_llm_stratified.sh \
              two_phase_run_multiview_split_stratified.sh; do
    echo "Running $script..."
    bash $script
    echo "Completed $script"
    echo ""
done
```

---

## 📝 结果收集脚本

创建一个自动收集分档结果的脚本：

```bash
#!/usr/bin/env bash
# collect_stratified_results.sh

echo "Collecting stratified evaluation results..."
echo ""

for log in log/*stratified*.log; do
    if [ -f "$log" ]; then
        echo "=== $(basename $log) ==="
        grep -E "Recall_(new|few|frequent)@10|NDCG_(new|few|frequent)@10|Coverage_(new|few|frequent)@10" "$log" | tail -9
        echo ""
    fi
done
```

---

## 🎯 预期结果模式

### 模式 1: 文本特征有效（预期）

```
               new      few      frequent
Recall@10:    高(0.05)  中(0.03)  低(0.02)
Coverage@10:  高(20%)   中(35%)   低(45%)
```
**解释**: 文本特征显著提升冷启动性能

### 模式 2: 文本特征无效（不希望）

```
               new      few      frequent
Recall@10:    低(0.02)  中(0.03)  高(0.04)
Coverage@10:  低(5%)    中(25%)   高(70%)
```
**解释**: 模型依赖交互历史，文本特征未起作用

### 模式 3: Per-item gate 有效（预期）

```
With text (threshold=5):
  Recall_frequent@10: 0.025  (gate=0, 不用文本)
  
Without text (baseline):
  Recall_frequent@10: 0.026  (差异很小)
  
→ Gate 成功将文本特征集中在 new/few items
```

---

## 📞 快速帮助

### 检查分档指标是否生效

```bash
# 运行任一脚本
bash two_phase_run_tfidf_stratified.sh

# 查看日志中是否有分档指标
grep "Recall_new" log/*.log
grep "Coverage_new" log/*.log

# 应该看到类似输出
```

### 调整分档阈值

编辑 `recbole/evaluator/stratified_metrics.py`:
```python
# 例如改为: new [1,5), few [5,20), frequent [20,+∞)
self.new_threshold = (1, 5)
self.few_threshold = (5, 20)
self.freq_threshold = (20, float('inf'))
```

---

**状态**: ✅ 完整实现，16个文件已创建  
**可执行**: 所有脚本已添加执行权限  
**下一步**: 选择数据集和模型，运行实验

