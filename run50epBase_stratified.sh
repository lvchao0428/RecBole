#!/usr/bin/env bash
# Pure SASRec baseline (70 epochs) with Stratified Evaluation
# Amazon_Beauty dataset

#cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "SASRec Baseline (70ep) with Stratified Metrics"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: Pure SASRec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlign \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_50ep_stratified.yaml" \
#  --nproc 8 

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

