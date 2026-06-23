#!/usr/bin/env bash
# Pure SASRec ID-only baseline (50ep) + stratified eval — book-crossing
# 与 run50epBase_food_stratified.sh 同协议：SASRecAlign + run_recbole.py

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
SEED="${SEED:-2025}"
GPU_ID="${GPU_ID:-0}"

echo "========================================="
echo "SASRec Baseline (50ep) — book-crossing, seed=$SEED"
echo "Model: SASRecAlign (pure ID, same as Beauty/Food)"
echo "========================================="

python run_recbole.py \
  --model SASRecAlign \
  --dataset book-crossing \
  --config_files "sasrec_baseline_50ep_book_crossing_stratified.yaml" \
  --config_dict "{'seed': ${SEED}, 'gpu_id': '${GPU_ID}', 'checkpoint_dir': 'saved/baseline_book_crossing_stratified_seed${SEED}'}"

echo "✅ Done"
