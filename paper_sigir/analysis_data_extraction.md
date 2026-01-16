# 数据整理报告

## 1. 敏感性分析数据提取 (Toys 7B Aggressive)

### Baseline (from line 63):
- Config: λ=0.10, τ=0.05, cold=2.5, infer=1.5
- HR@10: 6.92% (0.0692)
- NDCG@10: 4.51% (0.0451)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 1.99% (0.0199)

### Lambda (λ) Variations:
**λ=0.05** (line 78, exp_sensitivity_lambda_005):
- HR@10: 6.82% (0.0682)
- NDCG@10: 4.45% (0.0445)
- MRR@10: 3.72% (0.0372)
- HR_new@10: 1.98% (0.0198)

**λ=0.10** (baseline - line 63)
- HR@10: 6.92% (0.0692)
- NDCG@10: 4.51% (0.0451)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 1.99% (0.0199)

**λ=0.15** (line 79, exp_sensitivity_lambda_015):
- HR@10: 6.93% (0.0693)
- NDCG@10: 4.48% (0.0448)
- MRR@10: 3.72% (0.0372)
- HR_new@10: 2.04% (0.0204)

### Temperature (τ) Variations:
**τ=0.03** (line 80, exp_sensitivity_tau_003):
- HR@10: 6.94% (0.0694)
- NDCG@10: 4.50% (0.0450)
- MRR@10: 3.75% (0.0375)
- HR_new@10: 2.01% (0.0201)

**τ=0.05** (baseline - line 63)
- HR@10: 6.92% (0.0692)
- NDCG@10: 4.51% (0.0451)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 1.99% (0.0199)

**τ=0.10** (line 81, exp_sensitivity_tau_010):
- HR@10: 6.90% (0.0690)
- NDCG@10: 4.50% (0.0450)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 2.01% (0.0201)

### Cold-start Boost Variations:
**cold=1.5** (line 82, exp_sensitivity_cold_15):
- HR@10: 6.90% (0.0690)
- NDCG@10: 4.46% (0.0446)
- MRR@10: 3.71% (0.0371)
- HR_new@10: 1.95% (0.0195)

**cold=2.5** (baseline - line 63)
- HR@10: 6.92% (0.0692)
- NDCG@10: 4.51% (0.0451)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 1.99% (0.0199)

**cold=3.0** (line 89, exp_sensitivity_cold_30):
- HR@10: 6.89% (0.0689)
- NDCG@10: 4.47% (0.0447)
- MRR@10: 3.72% (0.0372)
- HR_new@10: 2.02% (0.0202)

### Inference Boost Variations:
**infer=0.5** (line 88, exp_sensitivity_infer_05):
- HR@10: 6.70% (0.0670)
- NDCG@10: 4.40% (0.0440)
- MRR@10: 3.69% (0.0369)
- HR_new@10: 1.95% (0.0195)

**infer=1.0** (需要查找，line 109标记为infer_10可能是1.0):
- HR@10: 6.85% (0.0685)
- NDCG@10: 4.46% (0.0446)
- MRR@10: 3.72% (0.0372)
- HR_new@10: 1.97% (0.0197)

**infer=1.5** (baseline - line 63)
- HR@10: 6.92% (0.0692)
- NDCG@10: 4.51% (0.0451)
- MRR@10: 3.76% (0.0376)
- HR_new@10: 1.99% (0.0199)

---

## 2. Seed实验数据提取

### Toys数据集:

#### TF-IDF:
- seed=42 (line 100): HR@10=6.51%, NDCG@10=4.35%, MRR@10=3.69%, HR_new@10=1.91%
- seed=2024 (line 103): HR@10=6.55%, NDCG@10=4.39%, MRR@10=3.72%, HR_new@10=1.82%
- seed=2025 (from line 24): HR@10=6.60%, NDCG@10=4.41%, MRR@10=3.73%, HR_new@10=1.83%

#### TF-IDF+LLM:
- seed=42 (line 101): HR@10=6.60%, NDCG@10=4.40%, MRR@10=3.72%, HR_new@10=1.91%
- seed=2024 (line 105): HR@10=6.59%, NDCG@10=4.41%, MRR@10=3.73%, HR_new@10=1.88%
- seed=2025 (from line 25): HR@10=6.66%, NDCG@10=4.41%, MRR@10=3.71%, HR_new@10=1.84%

#### MV-Align 7B:
- seed=42 (line 112): HR@10=6.66%, NDCG@10=4.38%, MRR@10=3.67%, HR_new@10=1.96%
- seed=2024 (line 106): HR@10=6.91%, NDCG@10=4.50%, MRR@10=3.76%, HR_new@10=1.92%
- seed=2025 (from line 26/63): HR@10=6.67% (standard), 6.92% (aggressive)

### Beauty数据集:

#### TF-IDF:
- seed=42 (line 107): HR@10=5.70%, NDCG@10=3.76%, MRR@10=3.16%, HR_new@10=1.63%
- seed=2024 (line 111): HR@10=5.66%, NDCG@10=3.74%, MRR@10=3.15%, HR_new@10=1.61%
- seed=2025 (from line 15): HR@10=5.63%, NDCG@10=3.73%, MRR@10=3.15%, HR_new@10=1.68%

#### TF-IDF+LLM:
- seed=42 (line 108): HR@10=5.76%, NDCG@10=3.79%, MRR@10=3.18%, HR_new@10=1.67%
- seed=2024 (line 113): HR@10=5.77%, NDCG@10=3.80%, MRR@10=3.19%, HR_new@10=1.67%
- seed=2025 (from line 16): HR@10=5.79%, NDCG@10=3.81%, MRR@10=3.20%, HR_new@10=1.63%

#### MV-Align 7B:
- seed=42 (line 110): HR@10=6.03%, NDCG@10=3.89%, MRR@10=3.24%, HR_new@10=1.75%
- seed=2024: **缺失数据** (需要运行 exp_seed2024_beauty_mv_7b.sh)
- seed=2025 (from line 17): HR@10=5.97%, NDCG@10=3.85%, MRR@10=3.20%, HR_new@10=1.67%

---

## 3. 数据完整性检查

### ✅ 完整的实验:
- Toys所有配置的seed实验 (42, 2024, 2025)
- Beauty的TF-IDF和TF-IDF+LLM的seed实验
- Toys的敏感性分析 (lambda, tau, cold, infer)

### ❌ 缺失的实验:
- Beauty seed=2024的MV-7B实验
- 敏感性分析的infer=1.0配置需要确认 (line 109可能不是标准的infer=1.0)

### ⚠️ 需要确认:
- CSV中敏感性分析标记为"beauty"但数值接近Toys baseline，需确认数据集
- line 83 (exp_sensitivity_cold_25) 标记为"doing"状态，可能未完成

