#!/usr/bin/env bash
# GTS Beauty ID-only Pilot V2 — eligible user filtering
# 1 seed (2025), min_train_interactions=5
#
# Usage (on log10):
#   nohup bash run_log10_ts_beauty_id_v2.sh \
#     > logs/ts_beauty_id_v2_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED=2025
MIN_TRAIN=5
LOG="logs/ts_beauty_id_v2.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS Beauty ID-only V2 (seed=$SEED, min_train=$MIN_TRAIN) on log10 ========"

t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_50ep_stratified_ts.yaml" \
  --config_dict "{'freeze_backbone': false}" \
  --gpu_id "$GPU_ID" \
  --min_train_interactions $MIN_TRAIN \
  --phase_a_epochs 50 \
  --phase_a_eval_step 5 \
  --phase_a_valid_metric "MRR@10" \
  --only_phase_a \
  --checkpoint_dir ./saved/ts_beauty_id_only_v2_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,id_only,beauty,ts,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_id_only_v2_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

log "======== ID-only V2 complete, syncing results to 5090 ========"
rsync -avz --include='*/' --include='*.pth' --include='*.json' --include='*.log' --exclude='*' \
  saved/ts_beauty_id_only_v2_seed${SEED}/ \
  charlie@192.168.0.106:/home/charlie/project/RecBole/saved/ts_beauty_id_only_v2_seed${SEED}/ \
  2>&1 | tee -a "$LOG"
log "  sync done"
