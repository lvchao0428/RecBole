#!/usr/bin/env bash
# Pure SASRec baseline (70 epochs) with Stratified Evaluation
# ML-1M dataset

cd /Users/lvchao0428/project/ownRecBole/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "SASRec Baseline (70ep) with Stratified Metrics (ML-1M)"
echo "========================================="
echo "Dataset: ml-1m"
echo "Model: Pure SASRec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlign \
  --dataset ml-1m \
  --config_files "sasrec_baseline_70ep_ml1m_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""

