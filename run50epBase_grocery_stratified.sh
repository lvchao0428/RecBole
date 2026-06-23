#!/usr/bin/env bash
# Pure SASRec ID-only baseline (50ep) + stratified eval — Grocery

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"
DS="Amazon_Grocery_and_Gourmet_Food"

python run_recbole.py \
  --model SASRecAlign \
  --dataset "$DS" \
  --config_files "sasrec_baseline_50ep_grocery_stratified.yaml" \
  --config_dict "{'seed': ${SEED}, 'gpu_id': '${GPU_ID}', 'checkpoint_dir': 'saved/baseline_grocery_stratified_seed${SEED}'}"

echo "✅ Done"
