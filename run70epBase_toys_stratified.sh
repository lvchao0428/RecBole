#!/usr/bin/env bash
# Pure SASRec baseline (70 epochs) with Stratified Evaluation
# Amazon_Toys_and_Games dataset

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "SASRec Baseline (70ep) with Stratified Metrics (Toys)"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "Model: Pure SASRec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlign \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_baseline_70ep_toys_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

