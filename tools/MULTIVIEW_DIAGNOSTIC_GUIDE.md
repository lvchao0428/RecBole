# Multi-View Diagnostic Guide

This guide provides tools and workflows to diagnose why multi-view performs differently on Beauty vs Toys datasets.

## Quick Start

### Option 1: Run Full Diagnostic Suite (Recommended)

```bash
cd /home/charlie/project/RecBole
bash tools/diagnose_multiview.sh
```

This will:
1. Analyze multi-view embedding quality for both datasets
2. Compare dataset characteristics
3. Analyze training logs (if available)
4. Generate a comprehensive summary report

**Output:** All results saved in `multiview_diagnostics_YYYYMMDD_HHMMSS/` directory

---

### Option 2: Run Individual Tools

#### 1. Analyze Multi-View Embedding Quality

Analyzes view diversity, feature space structure, and view-to-LLM similarity.

```bash
# For Beauty dataset
python tools/analyze_multiview_quality.py \
  --dataset Amazon_Beauty \
  --output beauty_analysis.json

# For Toys dataset
python tools/analyze_multiview_quality.py \
  --dataset Amazon_Toys_and_Games \
  --output toys_analysis.json
```

**Key metrics to look for:**
- `diversity_score`: Higher = views are more diverse (better)
- `effective_rank`: Higher = richer feature space (better)
- `mean_pairwise_cosine` (between views): Lower = more diverse (better)
- `view_vs_llm`: How similar each view is to single LLM

---

#### 2. Compare Dataset Characteristics

Compares interactions, text statistics, and embedding distributions.

```bash
python tools/compare_datasets.py
```

**Output:** `dataset_comparison_beauty_vs_toys.json`

**Key comparisons:**
- Inter-item similarity (if Toys items are too similar → harder to learn)
- Text length/word count (affects prompt effectiveness)
- Interaction density (affects alignment learning)

---

#### 3. Analyze Training Logs

Detects overfitting, convergence issues, alignment problems.

```bash
# Compare Beauty vs Toys logs
python tools/analyze_training_logs.py \
  --compare \
  --beauty_dir saved/phase_runs_multiview_4views \
  --toys_dir saved/phase_runs_multiview_4views_toys \
  --output log_comparison.json
```

**Key signals:**
- `overfitting_signal`: Large positive value = overfitting
- `improvement_trend`: Negative = performance degrading
- `avg_alignment_loss`: Compare between datasets

---

#### 4. Generate Tuning Suggestions

Automatically generates hyperparameter tuning recommendations.

```bash
python tools/generate_tuning_suggestions.py \
  --beauty_analysis beauty_analysis.json \
  --toys_analysis toys_analysis.json \
  --base_config sasrec_align_multi_view_toys.yaml \
  --output_dir config_variants
```

**Output:**
- `tuning_suggestions.json` - Detailed recommendations
- `config_variants/*.yaml` - Auto-generated config files to test

---

## Common Issues and Solutions

### Issue 1: Low View Diversity on Toys

**Symptoms:**
- `diversity_score` much lower than Beauty
- High `pairwise_cosine_similarity` between views

**Solutions:**
1. **Regenerate embeddings with more diverse prompts**
   ```bash
   # Edit tools/gen_multiview_4views_toys.sh
   # Try prompts that emphasize different aspects:
   # - Technical specifications vs emotional benefits
   # - Age-specific features vs general use cases
   # - Educational value vs entertainment value
   ```

2. **Increase SENet enhancement**
   ```yaml
   text_view_senet_ratio: 8  # Increase from 4
   ```

3. **Add view orthogonality regularization**
   ```yaml
   use_view_orthogonal_reg: true
   view_orthogonal_weight: 0.01
   ```

---

### Issue 2: High Inter-Item Similarity on Toys

**Symptoms:**
- `mean_pairwise_cosine` (LLM embeddings) much higher than Beauty
- `high_similarity_ratio` > 0.3

**Solutions:**
1. **Increase temperature for harder negatives**
   ```yaml
   temperature: 0.1  # Increase from 0.07
   ```

2. **Strengthen alignment**
   ```yaml
   alignment_weight: 0.1  # Increase from 0.05
   ```

3. **Add margin-based contrastive loss**
   ```yaml
   use_contrastive_margin: true
   contrastive_margin: 0.2
   ```

---

### Issue 3: Overfitting on Toys

**Symptoms:**
- Best validation performance occurs early
- Large `overfitting_signal` in log analysis
- Performance degrades after best epoch

**Solutions:**
1. **Increase regularization**
   ```yaml
   weight_decay: 5e-5          # Increase from 1e-5
   cross_dropout_prob: 0.6     # Increase from 0.5
   hidden_dropout_prob: 0.6    # Increase from 0.5
   ```

2. **Reduce model capacity**
   ```yaml
   text_cross_layer_num: 1     # Reduce from 2
   ```

3. **Early stopping**
   ```yaml
   stopping_step: 10           # Reduce from 20
   ```

---

### Issue 4: Low Effective Rank on Toys

**Symptoms:**
- `effective_rank` much lower than Beauty
- `singular_value_decay` close to 0

**Solutions:**
1. **Stronger whitening/normalization**
   ```yaml
   text_proj_norm: true
   fused_item_norm: true
   use_whitening: true
   whitening_strength: 0.8
   ```

2. **Match hidden size to intrinsic dimensionality**
   ```yaml
   hidden_size: 128  # Reduce from 256 if effective_rank << 128
   ```

---

## Recommended Workflow

### Step 1: Diagnose
```bash
# Run full diagnostic suite
bash tools/diagnose_multiview.sh

# Review results in multiview_diagnostics_*/
# Focus on: DIAGNOSIS_SUMMARY.md
```

### Step 2: Identify Root Cause

Compare key metrics between Beauty and Toys:

| Metric | Beauty | Toys | Issue? |
|--------|--------|------|--------|
| diversity_score | ? | ? | ✓ if Toys << Beauty |
| effective_rank | ? | ? | ✓ if Toys < 0.7 × Beauty |
| inter_item_cosine | ? | ? | ✓ if Toys > Beauty + 0.05 |
| overfitting_signal | ? | ? | ✓ if Toys > 0.01 |

### Step 3: Generate Solutions
```bash
python tools/generate_tuning_suggestions.py \
  --beauty_analysis multiview_diagnostics_*/multiview_quality_beauty.json \
  --toys_analysis multiview_diagnostics_*/multiview_quality_toys.json
```

### Step 4: Test Config Variants

Test auto-generated configs:
```bash
# Test variant 1 (addressing main issue)
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Toys_and_Games \
  --config_files "config_variants/variant_1_low_view_diversity.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.03,0.05" \
  --tau_grid "0.05,0.07,0.1" \
  ...

# Test combined variant
python scripts/two_phase_train.py \
  --config_files "config_variants/variant_combined_all_fixes.yaml" \
  ...
```

### Step 5: Iterate

Based on results:
1. Re-run diagnostics on new experiments
2. Compare to previous runs
3. Refine hyperparameters
4. Repeat until convergence

---

## Advanced Analysis

### Compute View-Specific Contribution

Check if certain views contribute more on Beauty than Toys:

```python
# In model code, log per-view attention/gate weights
# Then analyze:
import numpy as np

beauty_gates = [...]  # From logs
toys_gates = [...]

print(f"Beauty gate variance: {np.std(beauty_gates)}")
print(f"Toys gate variance: {np.std(toys_gates)}")
# Low variance on Toys = views contribute equally (might be redundant)
```

### Analyze View-Specific Alignment

```python
# Log per-view alignment losses
beauty_align = {'view1': [...], 'view2': [...], ...}
toys_align = {'view1': [...], 'view2': [...], ...}

# Compare which views align well
for view in beauty_align:
    b_loss = np.mean(beauty_align[view])
    t_loss = np.mean(toys_align[view])
    print(f"{view}: Beauty={b_loss:.4f}, Toys={t_loss:.4f}")
```

---

## Tool Reference

| Tool | Purpose | Key Output |
|------|---------|------------|
| `analyze_multiview_quality.py` | View diversity & feature analysis | diversity_score, effective_rank |
| `compare_datasets.py` | Dataset characteristic comparison | inter_item_similarity, text_stats |
| `analyze_training_logs.py` | Training dynamics analysis | overfitting_signal, convergence |
| `generate_tuning_suggestions.py` | Auto-generate fixes | config variants, recommendations |
| `diagnose_multiview.sh` | Run all tools at once | Comprehensive report |

---

## FAQ

**Q: Which tool should I run first?**  
A: Run `diagnose_multiview.sh` for a comprehensive analysis.

**Q: How do I know if my changes are working?**  
A: Re-run diagnostics after each experiment and compare:
- diversity_score should increase
- overfitting_signal should decrease
- Validation metrics should improve

**Q: What if none of the suggestions help?**  
A: Consider:
1. The views might be fundamentally too similar for Toys (try completely different prompts)
2. Single LLM might already be optimal for Toys (simpler is better)
3. Dataset characteristics might not favor multi-view learning

**Q: How can I visualize the differences?**  
A: Use the JSON outputs with:
```python
import json
import matplotlib.pyplot as plt

# Load and plot comparisons
with open('beauty_analysis.json') as f:
    beauty = json.load(f)
with open('toys_analysis.json') as f:
    toys = json.load(f)

# Plot diversity scores, effective ranks, etc.
```

---

## Contact & Support

If you encounter issues or need help interpreting results, check:
1. DIAGNOSIS_SUMMARY.md in the output directory
2. Individual JSON files for detailed metrics
3. Training logs for runtime issues

Good luck debugging! 🔍
