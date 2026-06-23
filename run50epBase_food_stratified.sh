#!/usr/bin/env bash
# Pure SASRec ID-only baseline (50ep) + stratified eval — Food

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"

echo "========================================="
echo "SASRec Baseline (50ep) — Food, seed=$SEED"
echo "========================================="

python run_recbole.py \
  --model SASRecAlign \
  --dataset Food \
  --config_files "sasrec_baseline_50ep_food_stratified.yaml" \
  --config_dict "{'seed': ${SEED}, 'gpu_id': '${GPU_ID}'}"

echo "✅ Done"
