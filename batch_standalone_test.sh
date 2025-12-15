#!/usr/bin/env bash
# Batch Standalone Test Script
# 
# Finds all model checkpoints in a directory and runs stratified evaluation on each.
#
# Usage:
#   ./batch_standalone_test.sh <checkpoint_directory> [output_directory]
#
# Example:
#   ./batch_standalone_test.sh saved/phase_runs_stratified results/stratified_tests

set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CHECKPOINT_DIR="${1:-saved}"
OUTPUT_DIR="${2:-results/standalone_tests}"

echo "========================================="
echo "Batch Standalone Test with Stratified Metrics"
echo "========================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all .pth files
CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pth" -type f 2>/dev/null | sort)

if [ -z "$CHECKPOINTS" ]; then
    echo "No .pth files found in $CHECKPOINT_DIR"
    exit 1
fi

COUNT=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')
echo "Found $COUNT checkpoint(s) to test"
echo ""

# Test each checkpoint
INDEX=0
for CHECKPOINT in $CHECKPOINTS; do
    INDEX=$((INDEX + 1))
    
    # Generate output filename from checkpoint path
    BASENAME=$(basename "$CHECKPOINT" .pth)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_stratified_results.txt"
    
    echo "========================================="
    echo "[$INDEX/$COUNT] Testing: $BASENAME"
    echo "========================================="
    
    python scripts/standalone_test.py \
        --model_file "$CHECKPOINT" \
        --metrics "Recall,MRR,NDCG,Hit,Precision,StratifiedRecall,StratifiedNDCG,StratifiedMRR,StratifiedHit,ItemPopularityStats" \
        --topk "5,10,20" \
        --output_file "$OUTPUT_FILE" \
        --no_progress \
        2>&1 | tee -a "$OUTPUT_DIR/batch_log.txt"
    
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
done

echo "========================================="
echo "✅ Batch Testing Complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary
echo "Generating summary..."
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
echo "Stratified Metrics Summary" > "$SUMMARY_FILE"
echo "==========================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for RESULT_FILE in "$OUTPUT_DIR"/*_stratified_results.txt; do
    if [ -f "$RESULT_FILE" ]; then
        BASENAME=$(basename "$RESULT_FILE" _stratified_results.txt)
        echo "### $BASENAME ###" >> "$SUMMARY_FILE"
        grep -E "(Recall_|NDCG_|MRR_|Hit_)(new|few|frequent)@10" "$RESULT_FILE" >> "$SUMMARY_FILE" || true
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "Summary saved to: $SUMMARY_FILE"

