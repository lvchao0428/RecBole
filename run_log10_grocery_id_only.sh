#!/usr/bin/env bash
# Grocery ID-only on log10 (1080Ti 11GB) — 小 batch 防 OOM

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/scripts/recbole_env.sh" ]]; then
  source "$ROOT/scripts/recbole_env.sh"
else
  export PATH="${HOME}/anaconda3/bin:${PATH}"
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"
DS="Amazon_Grocery_and_Gourmet_Food"
BATCH="${TRAIN_BATCH_SIZE:-256}"

echo "========================================="
echo "Grocery ID-only (50ep) on log10 — seed=$SEED batch=$BATCH"
echo "========================================="

python run_recbole.py \
  --model SASRecAlign \
  --dataset "$DS" \
  --config_files "sasrec_baseline_50ep_grocery_stratified.yaml" \
  --config_dict "{'seed': ${SEED}, 'gpu_id': '${GPU_ID}', 'train_batch_size': ${BATCH}, 'eval_batch_size': ${BATCH}, 'checkpoint_dir': 'saved/baseline_grocery_stratified_seed${SEED}'}"

echo "✅ Done"
