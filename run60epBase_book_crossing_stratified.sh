#!/usr/bin/env bash
# Pure SASRec baseline (60 epochs) with Stratified Evaluation
# 与 two_phase_*_book_crossing 中 Phase A 20 + Phase B 40 对齐（合计 60）
# book-crossing dataset（全量目录 dataset/book-crossing，无 -example）

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================="
echo "SASRec Baseline (60ep) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: Pure SASRec (no text features)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlign \
  --dataset book-crossing \
  --config_files "sasrec_baseline_60ep_book_crossing_stratified.yaml"

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""
