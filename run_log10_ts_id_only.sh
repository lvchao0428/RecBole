#!/usr/bin/env bash
# GTS ID-only baselines on log10 (1080Ti) — Beauty/Toys/Grocery × 2 seeds
# seed=2025 already done on 5090 for Beauty; need 2024, 42.
# Toys/Grocery need all 3 seeds.
#
# Usage (on log10 directly, or from 5090 via ssh):
#   nohup bash run_log10_ts_id_only.sh \
#     > logs/ts_id_only_log10_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/scripts/recbole_env.sh" ]] && source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
LOG="logs/ts_id_only_log10.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS ID-only on log10 ========"

RUN=0

run_id_only() {
  local DS="$1"
  local SEED="$2"
  local CONFIG="$3"
  RUN=$((RUN + 1))
  log ">>> [$RUN] ID-only ${DS} seed=${SEED}"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlign \
    --dataset "${DS}" \
    --config_files "${CONFIG}" \
    --config_dict "{'freeze_backbone': False}" \
    --gpu_id "$GPU_ID" \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "./saved/ts_${DS,,}_id_only_seed${SEED}" \
    --seed "$SEED" \
    --variant_features "sasrec,id_only,${DS,,},ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_${DS,,}_id_only_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

# Beauty: seeds 2024, 42 (seed=2025 already done on 5090)
for SEED in 2024 42; do
  run_id_only "Amazon_Beauty" "$SEED" "sasrec_baseline_50ep_stratified_ts.yaml"
done

# Toys: all 3 seeds
for SEED in 2024 2025 42; do
  run_id_only "Amazon_Toys_and_Games" "$SEED" "sasrec_baseline_50ep_stratified_toys_ts.yaml"
done

# Grocery: all 3 seeds
for SEED in 2024 2025 42; do
  run_id_only "Amazon_Grocery_and_Gourmet_Food" "$SEED" "sasrec_baseline_50ep_stratified_grocery_ts.yaml"
done

log "======== All ID-only runs complete ($RUN total) ========"
