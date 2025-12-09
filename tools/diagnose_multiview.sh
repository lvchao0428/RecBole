#!/usr/bin/env bash
#
# Multi-View Diagnosis Suite
#
# Runs comprehensive analysis to diagnose why multi-view works differently
# on Beauty vs Toys datasets.
#
# Usage:
#   bash tools/diagnose_multiview.sh

set -euo pipefail

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo ""
echo "================================================================================"
echo "MULTI-VIEW DIAGNOSTIC SUITE"
echo "================================================================================"
echo ""
echo "This suite will analyze:"
echo "  1. Multi-view embedding quality (diversity, feature space structure)"
echo "  2. Dataset characteristics (interactions, text statistics)"
echo "  3. Training logs (convergence, overfitting signals)"
echo ""
echo "Results will be saved in the current directory with detailed JSON reports."
echo ""
read -p "Press Enter to continue..."

# Create output directory
OUTPUT_DIR="multiview_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# 1. Analyze multi-view quality for both datasets
echo "================================================================================"
echo "1. ANALYZING MULTI-VIEW EMBEDDING QUALITY"
echo "================================================================================"
echo ""

echo "Analyzing Beauty dataset..."
python tools/analyze_multiview_quality.py \
  --dataset Amazon_Beauty \
  --output "$OUTPUT_DIR/multiview_quality_beauty.json" \
  2>&1 | tee "$OUTPUT_DIR/multiview_quality_beauty.log"

echo ""
echo "Analyzing Toys dataset..."
python tools/analyze_multiview_quality.py \
  --dataset Amazon_Toys_and_Games \
  --output "$OUTPUT_DIR/multiview_quality_toys.json" \
  2>&1 | tee "$OUTPUT_DIR/multiview_quality_toys.log"

# 2. Compare datasets
echo ""
echo "================================================================================"
echo "2. COMPARING DATASET CHARACTERISTICS"
echo "================================================================================"
echo ""

python tools/compare_datasets.py \
  2>&1 | tee "$OUTPUT_DIR/dataset_comparison.log"

# Move generated file to output dir
if [ -f "dataset_comparison_beauty_vs_toys.json" ]; then
  mv dataset_comparison_beauty_vs_toys.json "$OUTPUT_DIR/"
fi

# 3. Analyze training logs (if available)
echo ""
echo "================================================================================"
echo "3. ANALYZING TRAINING LOGS"
echo "================================================================================"
echo ""

if [ -d "saved/phase_runs_multiview_4views" ] || [ -d "saved/phase_runs_multiview_4views_toys" ]; then
  python tools/analyze_training_logs.py \
    --compare \
    --beauty_dir saved/phase_runs_multiview_4views \
    --toys_dir saved/phase_runs_multiview_4views_toys \
    --output "$OUTPUT_DIR/training_log_comparison.json" \
    2>&1 | tee "$OUTPUT_DIR/training_log_comparison.log"
else
  echo "⚠ No training logs found. Skipping log analysis."
  echo "  Expected directories:"
  echo "    - saved/phase_runs_multiview_4views"
  echo "    - saved/phase_runs_multiview_4views_toys"
fi

# 4. Generate summary report
echo ""
echo "================================================================================"
echo "4. GENERATING SUMMARY REPORT"
echo "================================================================================"
echo ""

SUMMARY_FILE="$OUTPUT_DIR/DIAGNOSIS_SUMMARY.md"

cat > "$SUMMARY_FILE" << 'EOF'
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
EOF

echo "✓ Summary report generated: $SUMMARY_FILE"

# Print summary
echo ""
echo "================================================================================"
echo "DIAGNOSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "All results saved to: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review $SUMMARY_FILE"
echo "  2. Check JSON files for detailed metrics"
echo "  3. Compare Beauty vs Toys key differences"
echo "  4. Adjust configuration based on findings"
echo ""
echo "================================================================================"
echo ""
