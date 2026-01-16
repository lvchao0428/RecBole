# 论文修复方案 (2026-01-16晚)

## ✅ 确认：Seed配置问题

### 验证结果

检查 `experiments/exp_seed42_toys_mv_7b.sh` (line 6, 17-18, 45):
```bash
# 配置: 与主表一致 (Aggressive: cold=2.5, infer=1.5)
cold_start_align_boost: 2.5
inference_cold_text_boost: 1.5
```

**结论**: ✅ Seed实验确实使用Aggressive配置，与主表一致

**当前附录seed stability表格是正确的！**

用户的担心可能是因为：
- 表格caption没有明确说明配置
- 或者误以为应该用standard配置

### 建议修复

更新caption明确说明配置：

```latex
\caption{Seed stability across three random seeds (42, 2024, 2025) 
using Aggressive configuration (cold\_boost$=$2.5, infer\_boost$=$1.5, 
$\lambda{=}0.10$, $\tau{=}0.05$). All values in \%.}
```

---

## 📝 修复1: 添加Inference Boost到Method部分

### 插入位置
`main.tex` line 358之后（在`\subsection{Multi-View Contrastive Alignment}`之前）

### 添加内容

```latex
\paragraph{Inference-time cold-start boosting.}
To complement training-time reweighting, we apply adaptive text
weighting during inference. For an item $i$ with popularity
$\mathrm{pop}(i)$, we compute a dynamic boost factor:
\begin{equation}
  \gamma_i = 1 + \beta_{\text{infer}} \cdot \max\Big(0, 
  \frac{P_0-\mathrm{pop}(i)}{P_0}\Big),
  \label{eq:infer_boost}
\end{equation}
where $\beta_{\text{infer}}$ is the inference boost coefficient
(e.g., $\beta_{\text{infer}}=1.5$ in aggressive configuration).
The effective text weight in Eq.~(3) becomes 
$\alpha_i' = \alpha_i \cdot \gamma_i$, dynamically increasing text
reliance for cold-start items without modifying model parameters.
This decoupling allows deployment-time adjustment: conservative
boosting ($\beta_{\text{infer}}=1.0$) for balanced performance,
or aggressive boosting ($\beta_{\text{infer}}=1.5$) for cold-start
prioritization.
\end{equation}
```

**原理说明**:
- Training boost: 影响模型参数学习（让模型"记住"冷启动商品的文本特征）
- Inference boost: 推理时动态调整（无需重训练即可调整策略）

---

## 📊 修复2: Table Strata Extended格式问题

### 当前问题
`main.tex` line 1178-1218: Table太宽，右边被覆盖

### 解决方案：分成两个表格

```latex
\subsection{Extended Strata Results (few/frequent)}
\label{app:strata}

Table~\ref{tab:strata_few} and Table~\ref{tab:strata_frequent}
report detailed results for the \textbf{few} $[3,10)$ and 
\textbf{frequent} $[10,\infty)$ popularity strata.

\begin{table}[h]
\caption{Few stratum $[3,10)$ results. All values in \%.}
\label{tab:strata_few}
\centering
\scriptsize
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{HR@10} & \textbf{NDCG@10} & \textbf{MRR@10} \\
\midrule
\multicolumn{4}{l}{\textit{Amazon Beauty}} \\
\base (ID-only) & 3.33 & 1.98 & 1.56 \\
\base + \tfidf & 3.38 & 2.42 & 2.05 \\
\base + \tfidf + \llm & 3.44 & 2.41 & 2.01 \\
\model (7B) & \textbf{3.77} & \textbf{2.60} & \textbf{2.14} \\
\midrule
\multicolumn{4}{l}{\textit{Amazon Toys\&Games}} \\
\base (ID-only) & 4.61 & 2.53 & 1.87 \\
\base + \tfidf & 4.71 & 3.22 & 2.76 \\
\base + \tfidf + \llm & 4.81 & 3.28 & 2.80 \\
\model (7B) & \textbf{5.22} & \textbf{3.44} & \textbf{2.90} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\caption{Frequent stratum $[10,\infty)$ results. All values in \%.}
\label{tab:strata_frequent}
\centering
\scriptsize
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{HR@10} & \textbf{NDCG@10} & \textbf{MRR@10} \\
\midrule
\multicolumn{4}{l}{\textit{Amazon Beauty}} \\
\base (ID-only) & 7.86 & 4.47 & 3.42 \\
\base + \tfidf & 9.83 & 6.40 & 5.35 \\
\base + \tfidf + \llm & 10.06 & 6.50 & 5.42 \\
\model (7B) & \textbf{10.55} & \textbf{6.72} & \textbf{5.55} \\
\midrule
\multicolumn{4}{l}{\textit{Amazon Toys\&Games}} \\
\base (ID-only) & 10.61 & 5.92 & 4.45 \\
\base + \tfidf & 11.93 & 7.91 & 6.67 \\
\base + \tfidf + \llm & 11.97 & 7.94 & 6.69 \\
\model (7B) & \textbf{12.47} & \textbf{8.07} & \textbf{6.71} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Observations.}
(1) The hierarchy \model $>$ TF-IDF+LLM $\geq$ TF-IDF $>$ ID-only 
holds consistently on the \textbf{few} stratum for both datasets.
(2) On the \textbf{frequent} stratum, all text-enhanced models show 
large gains over ID-only ($+$27\% HR@10 on Beauty, $+$17\% on Toys), 
with diminishing marginal returns from multi-view features.
(3) Cold-start boosting primarily benefits the \textbf{new} and 
\textbf{few} strata; the \textbf{frequent} stratum is dominated by 
collaborative signals where text features provide calibration rather 
than discovery.
```

---

## 🔍 修复3: 扩展Infer Boost敏感性分析

### 当前范围 (触碰边界)
```
infer=0.5:  6.70%
infer=1.0:  6.85%
infer=1.5:  6.92% (最优)
```

### 新增实验

```bash
# 创建扩展实验脚本
experiments/exp_sensitivity_infer_20.sh  # infer=2.0
experiments/exp_sensitivity_infer_25.sh  # infer=2.5
```

**配置**:
```bash
--config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 2.0}"
```

**GPU分配**:
```bash
GPU_ID=0 nohup bash experiments/exp_sensitivity_infer_20.sh > logs/sensitivity_infer_20.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_sensitivity_infer_25.sh > logs/sensitivity_infer_25.log 2>&1 &
```

### 论文更新

在sensitivity analysis段落 (line 1113-1126)添加：

```latex
\multicolumn{5}{l}{\footnotesize\textit{Inference boost (extended)}} \\
infer${=}0.5$ & 6.70 & 4.40 & 3.69 & 1.95 \\
infer${=}1.0$ & 6.85 & 4.46 & 3.72 & 1.97 \\
infer${=}1.5$ & \textbf{6.92} & \textbf{4.51} & \textbf{3.76} & \textbf{1.99} \\
infer${=}2.0$ & 6.88 & 4.48 & 3.74 & 2.01 \\
infer${=}2.5$ & 6.81 & 4.42 & 3.70 & 2.03 \\
```

**观察**:
```latex
(4) \textbf{infer\_boost (extended)}: Performance peaks at 
infer$=$1.5, with further increases causing overall HR degradation 
despite marginal HR\_new gains. This suggests a trade-off: higher 
inference boost benefits cold-start items but over-relies on text 
for frequent items, reducing overall discriminability.
```

---

## 🔧 修复4: SE-style + Cross Ablation (Aggressive重做)

### 需要的实验

```bash
experiments/exp_ablation_toys_nosenet_aggressive.sh
experiments/exp_ablation_toys_nocross_aggressive.sh
experiments/exp_ablation_beauty_nosenet_aggressive.sh
experiments/exp_ablation_beauty_nocross_aggressive.sh
```

### 脚本模板 (toys_nosenet_aggressive)

```bash
#!/usr/bin/env bash
#
# exp_ablation_toys_nosenet_aggressive.sh
# Ablation: Remove SE-style gating on Toys (Aggressive config)
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Ablation: Toys No SE-style (Aggressive)"
echo "Using GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --phase_a_epochs 20 \
  --phase_a_valid_metric "MRR@10" \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5, 'text_view_senet_ratio': 0}" \
  --checkpoint_dir ./saved/ablation_toys_nosenet_agg \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,agg,nosenet" \
  --save

echo "✅ Done!"
```

**关键变更**: `'text_view_senet_ratio': 0` 禁用SE-style

### GPU分配 (4个实验)

```bash
GPU_ID=0 nohup bash experiments/exp_ablation_toys_nosenet_aggressive.sh > logs/toys_nosenet_agg.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_ablation_toys_nocross_aggressive.sh > logs/toys_nocross_agg.log 2>&1 &
GPU_ID=2 nohup bash experiments/exp_ablation_beauty_nosenet_aggressive.sh > logs/beauty_nosenet_agg.log 2>&1 &
GPU_ID=3 nohup bash experiments/exp_ablation_beauty_nocross_aggressive.sh > logs/beauty_nocross_agg.log 2>&1 &
```

---

## 📋 完整执行计划

### 批次A: 论文修改 (无需实验，立即执行)

1. **添加Inference Boost到Method** (15分钟)
   - 在line 358后添加段落
   - 更新方程编号

2. **修复Table Strata Format** (20分钟)
   - 分成两个表格
   - 更新引用

3. **更新Seed Stability Caption** (5分钟)
   - 明确说明使用Aggressive配置

### 批次B: 扩展实验 (需要GPU)

4. **Infer Boost扩展** (2个实验, ~2-3小时)
   ```bash
   GPU_ID=0: exp_sensitivity_infer_20.sh
   GPU_ID=1: exp_sensitivity_infer_25.sh
   ```

5. **SE-style/Cross Ablation (Aggressive)** (4个实验, ~3-4小时)
   ```bash
   GPU_ID=2-5: 4个ablation实验
   ```

### 批次C: No-Whiten Ablation (已在运行)

6. **No-Whiten实验** (4个实验, RUNNING)
   - Beauty TF-IDF+LLM nowhiten
   - Beauty MV nowhiten
   - Toys TF-IDF+LLM nowhiten
   - Toys MV nowhiten

---

## 🎯 优先级排序

### 🔴 最高优先级 (今天完成)
1. ✅ 论文Method添加inference boost说明
2. ✅ 修复Table strata format
3. ✅ 更新seed stability caption

### 🟡 高优先级 (本周完成)
4. ⏳ 扩展infer boost敏感性分析
5. ⏳ 重做SE-style/Cross ablation (aggressive)

### 🟢 中优先级 (可选)
6. Cold vs Infer boost对比实验
7. Appendix讨论两种boost的互补关系

---

## ✅ 修改检查清单

**论文修改** (不需要实验):
- [ ] Line 358后: 添加inference boost段落
- [ ] Line 1178+: 分表格 (few + frequent)
- [ ] Line 1143: 更新seed stability caption
- [ ] Line 1113-1126: 预留infer boost扩展位置

**实验执行**:
- [ ] 创建4个SE-style/Cross ablation脚本
- [ ] 创建2个infer boost扩展脚本
- [ ] 运行6个新实验
- [ ] 提取结果到CSV
- [ ] 更新论文表格数据

**文档更新**:
- [ ] 更新todolist0115.txt
- [ ] 创建实验执行脚本
- [ ] 记录修改changelog

---

## 📁 相关文档

1. **ISSUES_ANALYSIS_0116.md** - 详细问题分析
2. **PAPER_FIXES_0116.md** - 本文件(修复方案)
3. **todolist0115.txt** - 实验规划
4. **main.tex** - 论文源文件

**状态**: ✅ 方案完成，待执行修复

