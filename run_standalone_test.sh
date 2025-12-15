#!/usr/bin/env bash
# Standalone Test Script for Stratified Metrics
# 
# Usage:
#   ./run_standalone_test.sh <model_checkpoint_path>
#
# Example:
#   ./run_standalone_test.sh saved/phase_runs_stratified/SASRecAlign-Amazon_Beauty-xxx.pth
#
# This script runs evaluation with the new stratified metrics:
#   - StratifiedRecall, StratifiedNDCG, StratifiedMRR, StratifiedHit
#   - Plus standard metrics: Recall, MRR, NDCG, Hit, Precision
#   - Coverage metrics: ItemPopularityStats

set -e

# Change to project root
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <model_checkpoint_path> [output_file]"
    echo ""
    echo "Example:"
    echo "  $0 saved/phase_runs_stratified/SASRecAlign-Amazon_Beauty-xxx.pth"
    echo "  $0 saved/model.pth results/test_results.txt"
    echo ""
    echo "Available checkpoints:"
    find saved -name "*.pth" -type f 2>/dev/null | head -20 || echo "  (no .pth files found in saved/)"
    exit 1
fi

MODEL_FILE="$1"
OUTPUT_FILE="${2:-}"

# Verify model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi

echo "========================================="
echo "Standalone Test with Stratified Metrics"
echo "========================================="
echo "Model: $MODEL_FILE"
echo ""
echo "Metrics to evaluate:"
echo "  Standard: Recall, MRR, NDCG, Hit, Precision"
echo "  Stratified: StratifiedRecall, StratifiedNDCG, StratifiedMRR, StratifiedHit"
echo "  Coverage: ItemPopularityStats"
echo ""
echo "Strata definitions:"
echo "  - new:      [1, 3)   training interactions"
echo "  - few:      [3, 10)  training interactions"
echo "  - frequent: [10, +∞) training interactions"
echo ""

# Build command
CMD="python scripts/standalone_test.py --model_file \"$MODEL_FILE\""
CMD="$CMD --metrics \"Recall,MRR,NDCG,Hit,Precision,StratifiedRecall,StratifiedNDCG,StratifiedMRR,StratifiedHit,ItemPopularityStats\""
CMD="$CMD --topk \"5,10,20\""

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi

# Run
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "========================================="
echo "✅ Evaluation Complete!"
echo "========================================="

