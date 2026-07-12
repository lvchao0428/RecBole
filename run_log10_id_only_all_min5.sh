#!/usr/bin/env bash
# log10 ID-only experiments — ALL datasets × 3 seeds, min5 filter
#
# Beauty/Toys/Grocery × seed=2024,2025,42 = 9 runs
# Each ~40 min on 1080Ti → ~6h total
#
# After completion, sync logs to 5090.
#
# Usage:
#   nohup bash run_log10_id_only_all_min5.sh > logs/log10_id_only_all_min5_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
LOG="logs/log10_id_only_all_min5.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_id_only() {
  local DATASET="$1" CONFIG="$2" SEED="$3" DS_SHORT="$4"
  local LOGFILE="logs/ts_${DS_SHORT}_id_only_min5_seed${SEED}.log"
  local CKPT="./saved/ts_${DS_SHORT}_id_only_min5_seed${SEED}"

  log "  >>> ${DATASET} ID-only min5 (seed=${SEED})"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlign \
    --dataset "$DATASET" \
    --config_files "$CONFIG" \
    --config_dict "{'freeze_backbone': false}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$CKPT" \
    --seed "$SEED" \
    --variant_features "id_only,ts,min5,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

log "======== log10 ID-only ALL datasets × 3 seeds (min5) ========"

SEEDS=(2024 2025 42)

# Beauty
log "=== Beauty ==="
for SEED in "${SEEDS[@]}"; do
  run_id_only Amazon_Beauty sasrec_baseline_50ep_stratified_ts.yaml "$SEED" beauty
done

# Toys
log "=== Toys ==="
for SEED in "${SEEDS[@]}"; do
  run_id_only Amazon_Toys_and_Games sasrec_baseline_50ep_stratified_toys_ts.yaml "$SEED" toys
done

# Grocery
log "=== Grocery ==="
for SEED in "${SEEDS[@]}"; do
  run_id_only Amazon_Grocery_and_Gourmet_Food sasrec_baseline_50ep_stratified_grocery_ts.yaml "$SEED" grocery
done

log "======== All ID-only experiments complete ========"

# Sync logs and results to 5090
log ">>> Syncing results to 5090 ..."
REMOTE_5090="charlie@192.168.0.106"
REMOTE_DIR="/home/charlie/project/RecBole"

rsync -avz logs/ts_*_id_only_min5_seed*.log \
  "${REMOTE_5090}:${REMOTE_DIR}/logs/log10/" \
  2>&1 | tee -a "$LOG"

rsync -avz "$LOG" \
  "${REMOTE_5090}:${REMOTE_DIR}/logs/log10/" \
  2>&1 | tee -a "$LOG"

log ">>> Sync complete"
log "======== Pipeline done ========"
