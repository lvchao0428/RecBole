# Multi-View Diagnosis Summary

## Overview

This report summarizes the diagnostic analysis of multi-view performance
on Beauty vs Toys datasets.

## Key Findings

### 1. Multi-View Embedding Quality

**Check the following files for details:**
- `multiview_quality_beauty.json` - Beauty dataset analysis
- `multiview_quality_toys.json` - Toys dataset analysis

**Key metrics to compare:**

1. **View Diversity Score** (higher is better)
   - Measures how different the views are from each other
   - Low diversity → views are redundant → less benefit from multi-view

2. **Effective Rank** (higher is better)
   - Indicates the intrinsic dimensionality of embeddings
   - Low rank → embeddings are in a low-dimensional subspace

3. **Mean Pairwise Cosine Similarity** (between views)
   - Lower similarity → more diverse views → better multi-view potential

4. **View vs Single LLM Similarity**
   - If views are too similar to single LLM → multi-view adds little value

### 2. Dataset Characteristics

**Check:** `dataset_comparison_beauty_vs_toys.json`

**Key comparisons:**

1. **Inter-item Similarity**
   - If Toys items are more similar → harder to distinguish → less benefit
   
2. **Text Statistics**
   - Title length, word count differences
   - May affect how well different prompts can extract diverse views

3. **Interaction Patterns**
   - Density, sparsity differences
   - May affect alignment learning

### 3. Training Dynamics

**Check:** `training_log_comparison.json`

**Look for:**

1. **Overfitting signals**
   - Does Toys overfit earlier than Beauty?
   
2. **Alignment loss patterns**
   - Is alignment loss behaving differently?
   
3. **Convergence speed**
   - Does Toys converge too fast (underfitting) or too slow?

## Recommended Actions

Based on the findings above, consider:

### If View Diversity is Low on Toys:
- Try different prompts to increase view diversity
- Adjust per-view weights (give more weight to diverse views)
- Increase SENet compression ratio for stronger view enhancement

### If Items are Too Similar on Toys:
- Increase regularization (weight_decay, dropout)
- Try stronger text feature normalization
- Consider temperature scheduling for alignment

### If Overfitting is Detected:
- Reduce model capacity (smaller cross network)
- Increase dropout rates
- Early stopping based on validation metrics

### If Underfitting is Detected:
- Increase model capacity
- Reduce regularization
- Train longer (more epochs)

### If Alignment is Problematic:
- Grid search over alignment_weight (try 0.01, 0.03, 0.05, 0.1)
- Try different temperature values
- Consider per-view alignment weights

## Next Steps

1. Review all JSON files in this directory for detailed metrics
2. Compare key metrics between Beauty and Toys
3. Identify the main differences
4. Adjust configuration based on findings
5. Re-run experiments with updated settings

---

**Generated:** $(date)
