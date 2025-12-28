#!/usr/bin/env bash
# Pure BERT4Rec baseline (50 epochs) with Stratified Evaluation
# Amazon_Beauty dataset

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "BERT4Rec Baseline (50ep) with Stratified Metrics"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: Pure BERT4Rec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model BERT4RecAlign \
  --dataset Amazon_Beauty \
  --config_files "bert4rec_baseline_50ep_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

