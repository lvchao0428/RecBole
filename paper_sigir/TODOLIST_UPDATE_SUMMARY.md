# Todolist 0115 更新总结

## 📝 更新内容

### 1. 批次 2: No-Whiten Ablation (新增)

**位置**: todolist0115.txt lines 74-97

**更新说明**:
- ✅ 确认脚本已改为Aggressive配置 (cold=2.5, infer=1.5)
- ✅ 添加详细的实验说明和预期结果
- ✅ 更新GPU分配方案 (4个实验使用GPU 0-3)

**实验脚本** (已确认为Aggressive配置):
```bash
beauty_train/two_phase_run_tfidf_llm_stratified_no_whiten.sh
beauty_train/two_phase_run_multiview_v2_stratified_no_whiten.sh
toy_train/two_phase_run_tfidf_llm_toys_stratified_no_whiten.sh
toy_train/two_phase_run_multiview_v2_toys_stratified_7b_no_whiten.sh
```

**目的**:
- 补全论文Table ablation_whiten (main.tex line 784-802)
- 添加Beauty的no-whiten数据，形成完整对比
- 验证whitening在aggressive配置下的效果

---

### 2. 批次 3: 补充Ablation实验 (新增)

**位置**: todolist0115.txt lines 150-217

**包含实验**:

#### Tier 1 - 必须做 (3个):
1. **Beauty No-Cross** - 补全Table ablation_senet_cross
2. **Cold-start Reweighting** - 兑现Appendix承诺
3. **Center-only Normalization** - 兑现Appendix承诺

#### Tier 2 - 强烈推荐 (4个):
4. **TF-IDF/TF-IDF+LLM with Boost** - 修复层级反转问题

---

### 3. 实验优先级建议 (新增)

**位置**: todolist0115.txt lines 192-217

**优先级排序**:

🔴 **最高优先级** (立即执行):
- 批次2: No-Whiten Ablation (4个实验)
- 原因: Beauty数据缺失，论文表格不完整

🟡 **高优先级** (本轮完成):
- Beauty No-Cross
- Cold-start Reweighting
- Center-only Normalization
- 原因: 兑现Appendix承诺，提升论文完整度

🟢 **中优先级** (可选):
- TF-IDF with Boost系列
- 原因: 修复层级反转，确保fair comparison

---

## 🚀 快速启动

### 启动批次2 (No-Whiten Ablation):

**使用专用脚本** (推荐):
```bash
cd /Users/charlie.lyu/project/RecBole
bash paper_sigir/run_no_whiten_ablation.sh
```

**或手动启动**:
```bash
# Beauty (2个)
GPU_ID=0 nohup bash beauty_train/two_phase_run_tfidf_llm_stratified_no_whiten.sh > logs/beauty_tfidf_llm_no_whiten.log 2>&1 &
GPU_ID=1 nohup bash beauty_train/two_phase_run_multiview_v2_stratified_no_whiten.sh > logs/beauty_mv_no_whiten.log 2>&1 &

# Toys (2个)
GPU_ID=2 nohup bash toy_train/two_phase_run_tfidf_llm_toys_stratified_no_whiten.sh > logs/toys_tfidf_llm_no_whiten.log 2>&1 &
GPU_ID=3 nohup bash toy_train/two_phase_run_multiview_v2_toys_stratified_7b_no_whiten.sh > logs/toys_mv_no_whiten.log 2>&1 &
```

### 启动批次3 (其他Ablation):
```bash
# Tier 1 (3个)
GPU_ID=4 nohup bash experiments/exp_ablation_beauty_nocross.sh > logs/ablation_beauty_nocross.log 2>&1 &
GPU_ID=5 nohup bash experiments/exp_ablation_no_cold_reweight_toys.sh > logs/ablation_no_cold_reweight.log 2>&1 &
GPU_ID=6 nohup bash experiments/exp_ablation_center_only_toys.sh > logs/ablation_center_only.log 2>&1 &

# Tier 2 (按需)
GPU_ID=7 nohup bash experiments/exp_tfidf_toys_with_boost.sh > logs/tfidf_toys_boost.log 2>&1 &
```

---

## 📊 实验资源规划

### GPU分配:
- **5090** (1张): 可用于批次3
- **4090** (8张):
  - GPU 0-3: 批次2 (no-whiten, 4个)
  - GPU 4-7: 批次3 (ablation, 4-7个)

### 时间估计:
- 批次2: ~2-3小时 (4个实验)
- 批次3 Tier 1: ~2-3小时 (3个实验)
- 批次3 Tier 2: ~2-3小时 (4个实验)

**总计**: 如果顺序执行，约6-9小时；如果并行执行，约2-3小时

---

## 📋 实验完成后的论文更新

### 批次2完成后:

#### 更新 Table ablation_whiten (line 784-802):

**当前表格** (仅Toys):
```latex
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Config} & HR@10 & NDCG@10 & MRR@10 & HR\_new@10 \\
\midrule
\multicolumn{5}{l}{\textbf{Multi-view 7B}} \\
w/ Whiten & 6.92 & 4.51 & 3.76 & 1.99 \\
w/o Whiten & 6.86 & 4.45 & 3.70 & 1.99 \\
\midrule
\multicolumn{5}{l}{\textbf{Single-view (TF-IDF+LLM)}} \\
w/ Whiten & 6.66 & 4.41 & 3.71 & 1.84 \\
w/o Whiten & 6.75 & 4.49 & 3.75 & 1.84 \\
\bottomrule
\end{tabular}
```

**更新后** (添加Beauty):
```latex
\begin{tabular}{@{}llcccc@{}}
\toprule
\textbf{Dataset} & \textbf{Config} & HR@10 & NDCG@10 & MRR@10 & HR\_new@10 \\
\midrule
\multicolumn{6}{l}{\textbf{Multi-view 7B}} \\
Beauty & w/ Whiten & 6.07 & 3.93 & 3.27 & 1.81 \\
Beauty & w/o Whiten & ?.?? & ?.?? & ?.?? & ?.?? \\
Toys & w/ Whiten & 6.92 & 4.51 & 3.76 & 1.99 \\
Toys & w/o Whiten & 6.86 & 4.45 & 3.70 & 1.99 \\
\midrule
\multicolumn{6}{l}{\textbf{Single-view (TF-IDF+LLM)}} \\
Beauty & w/ Whiten & 5.74 & 3.77 & 3.17 & 1.65 \\
Beauty & w/o Whiten & ?.?? & ?.?? & ?.?? & ?.?? \\
Toys & w/ Whiten & 6.66 & 4.41 & 3.71 & 1.84 \\
Toys & w/o Whiten & 6.75 & 4.49 & 3.75 & 1.84 \\
\bottomrule
\end{tabular}
```

---

### 批次3完成后:

#### 1. 更新 Table ablation_senet_cross (line 732-751)
添加Beauty的no-cross数据

#### 2. 补充 Appendix Section (line 1171-1177)
添加cold-start reweighting和center-only normalization的完整分析

#### 3. (可选) 更新 Main Table 或 Discussion
如果boost实验显著，讨论fair comparison问题

---

## ✅ 检查清单

### Todolist更新:
- [x] 批次2: No-Whiten Ablation说明
- [x] 确认脚本路径和配置
- [x] 批次3: 其他Ablation实验
- [x] 实验优先级建议
- [x] GPU分配方案

### 执行脚本:
- [x] 创建 `run_no_whiten_ablation.sh`
- [x] 添加监控和状态检查
- [x] 添加预期结果说明

### 文档:
- [x] 创建本更新总结
- [x] 关联到 `proposed_experiments.md`
- [x] 关联到 `TASK_COMPLETE_0116.md`

---

## 📎 相关文档

1. **todolist0115.txt** - 主要实验规划文档 (已更新)
2. **run_no_whiten_ablation.sh** - 批次2专用启动脚本 (新建)
3. **proposed_experiments.md** - 详细实验方案
4. **experiment_batch_commands.sh** - 原5个核心实验脚本
5. **TASK_COMPLETE_0116.md** - 任务完成总结

---

## 🎯 下一步行动

### 立即执行:
```bash
cd /Users/charlie.lyu/project/RecBole
bash paper_sigir/run_no_whiten_ablation.sh
```

### 等待完成后:
1. 提取实验结果到CSV
2. 更新论文Table ablation_whiten
3. 启动批次3实验
4. 最终更新论文 → 投稿

---

**更新时间**: 2026-01-16  
**更新内容**: 批次2+3实验规划，优先级建议  
**状态**: ✅ Ready for execution  

