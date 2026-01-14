# 剩余实验规划 (2026-01-14)

## ✅ 当前运行中的实验

| GPU | 实验 | 配置 | 状态 | 预计完成 |
|-----|------|------|------|----------|
| 5090-0 | cold_25 | cold=2.5 baseline重跑 | 🔄 运行中 | ~1小时 |
| 4090-1 | infer_05 | infer=0.5 | 🔄 运行中 | ~2小时 |
| 4090-2 | cold_30 | cold=3.0 | 🔄 运行中 | ~2小时 |
| 4090-3 | lambda_005 | λ=0.05 | ✅ 完成 | - |
| 4090-4 | lambda_015 | λ=0.15 | ✅ 完成 | - |
| 4090-5 | tau_003 | τ=0.03 | ✅ 完成 | - |
| 4090-6 | tau_010 | τ=0.10 | ✅ 完成 | - |
| 4090-7 | cold_15 | cold=1.5 | ✅ 完成 | - |

## 📋 待补充的实验

### 1. ⭐⭐⭐ Uni100 vs Full-Ranking 对比实验 (高优先级)

**目的**: 验证 sampled evaluation (uni100) 与 full-ranking 的一致性

**论文位置**: Appendix A.1 - Table "Sampled Evaluation Sanity Check"

**需要的实验** (✅ 脚本已全部创建):

```bash
# Toys 7B Aggressive - Uni100 评估 (完整二阶段训练)
GPU_ID=3 nohup bash experiments/exp_uni100_toys_7b.sh > uni100_toys_7b.log 2>&1 &  # ✅

# Toys TF-IDF+LLM - Uni100 评估 (完整二阶段训练)
GPU_ID=4 nohup bash experiments/exp_uni100_tfidf_llm_toys.sh > uni100_tfidf_llm_toys.log 2>&1 &  # ✅

# Toys TF-IDF - Uni100 评估 (完整二阶段训练)
GPU_ID=5 nohup bash experiments/exp_uni100_tfidf_toys.sh > uni100_tfidf_toys.log 2>&1 &  # ✅
```

**对比数据** (Full-ranking 已有):
| Model | Full HR@10 | Full MRR@10 | Uni100 HR@10 | Uni100 MRR@10 | Correlation |
|-------|-----------|-------------|--------------|---------------|-------------|
| Multi-view 7B | 6.92 | 3.76 | ? | ? | ? |
| TF-IDF+LLM | 6.66 | 3.71 | ? | ? | ? |
| TF-IDF | 6.60 | 3.73 | ? | ? | ? |

**分析指标**:
- Overall correlation (Pearson/Spearman)
- Per-stratum correlation (new/few/frequent)
- Model mis-ranking cases

---

### 2. ⭐⭐ Scale Law 补充实验 (中优先级)

**目的**: 完整验证 Toys Standard 配置的 Scale Law

**论文位置**: Appendix A.2 - Scale Law Validation

**需要的实验**:

```bash
# Toys 14B Standard (cold=2.0, infer=1.0)
GPU_ID=3 nohup bash two_phase_run_multiview_v2_toys_stratified_14b_standard.sh > toys_14b_std.log 2>&1 &

# Toys 32B Standard (cold=2.0, infer=1.0)  
GPU_ID=4 nohup bash two_phase_run_multiview_v2_toys_stratified_32b_standard.sh > toys_32b_std.log 2>&1 &
```

**已有**: Toys 7B Standard (row 70, 0111.csv)

**预期 Scale Law**:
- HR@10: 32B > 14B > 7B
- HR_new@10: 32B > 14B > 7B
- MRR@10: 可能反转（附录中说明）

---

### 3. ⭐ Beauty 补充实验 (低优先级)

**目的**: Beauty 数据集的完整对比

**可选实验**:

```bash
# Beauty TF-IDF baseline (已有数据，可跳过)
# Beauty TF-IDF+LLM (已有数据，可跳过)
# Beauty Multi-view 7B Aggressive (如需统一配置)
```

---

## 🎯 推荐实验优先级

### 高优先级 (必须完成)

1. **Uni100 对比实验** (3个实验，约6小时)
   - 论文 Appendix 必需
   - 展示 full-ranking 的必要性

### 中优先级 (建议完成)

2. **Toys Standard 14B/32B** (2个实验，约4小时)
   - 完整验证 Scale Law
   - 增强论文说服力

### 低优先级 (可选)

3. **Beauty 补充实验**
   - 如果审稿人要求再补充

---

## 📝 实验脚本创建清单

需要创建的脚本：

- [ ] `experiments/exp_uni100_toys_7b.sh`
- [ ] `experiments/exp_uni100_tfidf_llm_toys.sh`
- [ ] `experiments/exp_uni100_tfidf_toys.sh`
- [ ] `two_phase_run_multiview_v2_toys_stratified_14b_standard.sh`
- [ ] `two_phase_run_multiview_v2_toys_stratified_32b_standard.sh`

---

## 🔬 Whiten 方案真实情况 (重要发现！)

### 从脚本发现：

**所有特征都使用 `--center --whiten` (ZCA whitening)：**

1. **TF-IDF**: `--center --whiten` ✅
2. **Single-view LLM**: `--center --whiten` ✅
3. **Multi-view (per-view)**: `--center --whiten` ✅

### 这意味着：

- ❌ **没有 "no ZCA" 的情况**
- ✅ 所有文本特征都经过: Center → ZCA Whiten → L2
- ✅ 差异在于: Multi-view 是 **per-view whiten** (每个视图独立白化)

### 架构图需要修正：

**Single-view:**
```
TF-IDF → SVD → Whiten (ZCA) → SENet → L2
LLM    → SVD → Whiten (ZCA) → SENet → L2
```

**Multi-view:**
```
TF-IDF base → SVD → Whiten (ZCA, shared) → SENet → L2
View 1-4    → SVD → Whiten (ZCA, per-view) → SENet → L2
```

**关键差异**: 
- Single-view: 所有特征用相同的 whiten matrix
- Multi-view: 每个视图用独立的 whiten matrix (4+1=5个)

---

## 🎯 建议行动

### 立即执行 (等当前3个实验完成后)

1. **Uni100 对比实验** (填充 Appendix Table)
2. **更新架构图** (修正 whiten 标注)

### 次要

3. **Standard 配置 Scale Law** (如果需要完整对比)

---

## 📊 论文完成度

| 部分 | 进度 | 状态 |
|------|------|------|
| Main Results (Table 3) | 100% | ✅ 完成 |
| Ablation (Table 4, 5) | 100% | ✅ 完成 |
| Sensitivity (Table 10) | 75% | 🔄 运行中 |
| Uni100 Sanity Check (App Table) | 0% | ⏳ 待开始 |
| Scale Law (App Table) | 33% | ⏳ 可选 |

**预计投稿时间**: 完成 Uni100 实验后 (约1周)
