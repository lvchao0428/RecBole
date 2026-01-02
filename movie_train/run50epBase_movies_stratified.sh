#!/usr/bin/env bash
# Pure SASRec baseline (70 epochs) with Stratified Evaluation
# Amazon_Movies_and_TV dataset

#cd /Users/lvchao0428/project/ownRecBole/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "SASRec Baseline (70ep) with Stratified Metrics (Movies)"
echo "========================================="
echo "Dataset: Amazon_Movies_and_TV"
echo "Model: Pure SASRec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlign \
  --dataset Amazon_Movies_and_TV \
  --config_files "sasrec_baseline_70ep_movies_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

