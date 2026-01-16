# 5个新实验建议 - 完善论文Ablation

## 📊 当前论文gaps分析

### 已有的ablation (从CSV):
- ✅ Toys: no-whiten, no-cross, no-senet
- ✅ Beauty: no-senet
- ❌ Beauty: **缺少 no-whiten, no-cross**

### 论文Appendix明确提到但未实现:
- ❌ Cold-start reweighting ablation (line 1173)
- ❌ Center-only normalization (vs ZCA whitening) (line 1175)

### 其他重要gaps:
- 仅Toys有完整ablation，Beauty数据不完整
- Whitening方法对比 (ZCA vs center-only vs none)
- Alignment objective的详细ablation

---

## 🎯 推荐的5个实验 (优先级排序)

### **实验1: Beauty No-Whiten (高优先级)** ⭐⭐⭐
**动机**: 补全Beauty的ablation数据，与Toys对应
**配置**: MV-7B on Beauty, Aggressive, no whitening
**预期**: 验证whitening在Beauty上的效果是否与Toys一致

**实验脚本**:
```bash
experiments/exp_ablation_beauty_nowhiten.sh
```

**论文更新位置**: Table~\ref{tab:ablation_whiten} (line 784-802)
- 添加Beauty部分的no-whiten数据

**预期结果**:
- 如果与Toys一致: no-whiten会降低MV性能
- 可以在论文中claim: "whitening对multi-view的益处在两个数据集上consistent"

---

### **实验2: Beauty No-Cross (高优先级)** ⭐⭐⭐
**动机**: 补全Beauty的cross network ablation
**配置**: MV-7B on Beauty, Aggressive, no cross network
**预期**: 验证cross network的HR-MRR trade-off在Beauty上是否成立

**实验脚本**:
```bash
experiments/exp_ablation_beauty_nocross.sh
```

**论文更新位置**: Table~\ref{tab:ablation_senet_cross} (line 732-751)
- 添加Beauty部分，形成完整对比

**当前表格** (line 744-750):
```latex
Full (SE-style + Cross) & 6.92 & 4.51 & 3.76 & 1.99 \\
$-$ SE-style & 6.97 & 4.54 & 3.78 & 1.99 \\
$-$ Cross & 7.45 & 4.20 & 3.18 & 2.11 \\
```

**补充后**:
```latex
\midrule
\multicolumn{5}{l}{\textit{Amazon Beauty}} \\
Full (SE-style + Cross) & 6.07 & 3.93 & 3.27 & 1.81 \\
$-$ Cross & ? & ? & ? & ? \\
```

---

### **实验3: Cold-start Reweighting Ablation (高优先级)** ⭐⭐⭐
**动机**: Appendix line 1173明确承诺的实验
**配置**: 
- MV-7B on Toys, Aggressive
- 设置 cold_start_align_boost = 0.0 (关闭cold-start reweighting)
**预期**: 验证cold-start reweighting对new/few strata的影响

**实验脚本**:
```bash
experiments/exp_ablation_no_cold_reweight_toys.sh
```

**论文更新位置**: Appendix~\ref{app:additional} (line 1171-1177)
- 补充promised的cold-start reweighting ablation

**分析要点**:
- 对比 HR_new@10 和 HR_few@10
- 验证cold-start boosting是否essential for tail performance

---

### **实验4: Center-only Normalization (中优先级)** ⭐⭐
**动机**: Appendix line 1175承诺，对比不同normalization方法
**配置**: 
- MV-7B on Toys, Aggressive
- 使用 center-only (zero-mean) 而非 ZCA whitening
**预期**: 验证decorrelation (ZCA) vs centering-only的效果差异

**实验脚本**:
```bash
experiments/exp_ablation_center_only_toys.sh
```

**论文更新位置**: Appendix~\ref{app:additional}
- 添加normalization方法对比表格

**理论意义**:
- ZCA = centering + decorrelation + rescaling
- Center-only = 仅zero-mean
- 可以分解whitening的贡献: decorrelation的价值

---

### **实验5: TF-IDF Baseline with Cold-start Boost (中优先级)** ⭐⭐
**动机**: 修复论文中提到的"层级反转"问题
**配置**:
- TF-IDF on Toys/Beauty, Aggressive
- 添加 cold_boost=2.5, infer_boost=1.5 (与MV相同)
**预期**: HR_new@10 应该超过ID-only baseline

**背景** (从todolist line 106-131):
```
问题: Beauty/Toys 的部分指标出现层级反转:
  - TF-IDF HR_new < ID-only (Toys: 1.83% < 1.91%)
  - TF-IDF+LLM ≈ TF-IDF (差异小于 0.5%)

原因: TF-IDF/TF-IDF+LLM 未使用 cold_boost 和 infer_boost
```

**实验脚本**:
```bash
experiments/exp_tfidf_toys_with_boost.sh
experiments/exp_tfidf_llm_toys_with_boost.sh
experiments/exp_tfidf_beauty_with_boost.sh
experiments/exp_tfidf_llm_beauty_with_boost.sh
```

**论文更新位置**: 
- 可以添加到main table或者作为fair comparison实验
- 或者在Discussion中说明baseline的boost配置差异

**重要性**:
- 确保fair comparison (所有模型使用相同的boost策略)
- 或者在论文中明确说明boost是MV特有的优化

---

## 📋 实验优先级总结

### Tier 1 - 必须做 (论文完整性):
1. ✅ **Beauty No-Whiten** - 补全Table ablation_whiten
2. ✅ **Beauty No-Cross** - 补全Table ablation_senet_cross
3. ✅ **Cold-start Reweighting** - 兑现Appendix承诺

### Tier 2 - 强烈推荐 (提升论文质量):
4. ✅ **Center-only Normalization** - 兑现Appendix承诺
5. ✅ **TF-IDF with Boost** - 修复层级反转问题

### Tier 3 - 可选 (如有时间):
6. Beauty的no-senet-nocross组合实验
7. 不同temperature的更细粒度分析
8. Per-view alignment weight的ablation

---

## 🖥️ GPU分配建议 (5个实验)

基于用户提供的资源: **5090 1张卡 + 4090 8张卡**

### 方案A: 并行运行5个实验 (推荐)
```bash
# 5090 (1张)
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nowhiten.sh > ablation_beauty_nowhiten.log 2>&1 &

# 4090 (4张)
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nocross.sh > ablation_beauty_nocross.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_ablation_no_cold_reweight_toys.sh > ablation_no_cold_reweight.log 2>&1 &
GPU_ID=2 nohup bash experiments/exp_ablation_center_only_toys.sh > ablation_center_only.log 2>&1 &
GPU_ID=3 nohup bash experiments/exp_tfidf_toys_with_boost.sh > tfidf_toys_boost.log 2>&1 &
```

**预计完成时间**: ~2-3小时

### 方案B: 如果要跑所有boost实验 (8个实验)
```bash
# 5090
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nowhiten.sh > ablation_beauty_nowhiten.log 2>&1 &

# 4090 (7张)
GPU_ID=0 nohup bash experiments/exp_ablation_beauty_nocross.sh > ablation_beauty_nocross.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_ablation_no_cold_reweight_toys.sh > ablation_no_cold_reweight.log 2>&1 &
GPU_ID=2 nohup bash experiments/exp_ablation_center_only_toys.sh > ablation_center_only.log 2>&1 &
GPU_ID=3 nohup bash experiments/exp_tfidf_toys_with_boost.sh > tfidf_toys_boost.log 2>&1 &
GPU_ID=4 nohup bash experiments/exp_tfidf_llm_toys_with_boost.sh > tfidf_llm_toys_boost.log 2>&1 &
GPU_ID=5 nohup bash experiments/exp_tfidf_beauty_with_boost.sh > tfidf_beauty_boost.log 2>&1 &
GPU_ID=6 nohup bash experiments/exp_tfidf_llm_beauty_with_boost.sh > tfidf_llm_beauty_boost.log 2>&1 &
```

---

## 📝 论文修改计划

### 主体部分更新:
1. **Table ablation_whiten** (line 784-802)
   - 添加Beauty的multi-view no-whiten数据
   
2. **Table ablation_senet_cross** (line 732-751)
   - 扩展为双数据集对比表格
   - 添加Beauty的ablation数据

### Appendix更新:
3. **Section Additional Ablations** (line 1171-1177)
   - 补充cold-start reweighting ablation (带表格和分析)
   - 补充center-only normalization ablation (带表格和分析)

### 可选更新:
4. 如果boost实验结果显著:
   - 在Discussion中添加一段fair comparison的讨论
   - 或者更新main table添加"fair baseline"行

---

## ✅ 预期论文提升

完成这5个实验后:
- ✅ 兑现所有Appendix承诺
- ✅ Beauty和Toys的ablation数据对称完整
- ✅ 增强论文的robustness claims
- ✅ 修复潜在的reviewer concern (层级反转, fair comparison)
- ✅ 提供更深入的component analysis

**预计reviewer满意度**: ⭐⭐⭐⭐⭐ → 论文更solid, 更难被reject

