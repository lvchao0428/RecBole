#!/usr/bin/env bash
# GTS UniSRec Beauty — 3 configs × 3 seeds = 9 runs on 5090
#
# Usage (on 5090):
#   nohup bash run_5090_ts_unisrec.sh \
#     > logs/ts_unisrec_5090_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
LOG="logs/ts_unisrec_5090.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

SEEDS=(2024 2025 42)
RUN=0
TOTAL=9

log "======== GTS UniSRec Beauty: 3 configs × 3 seeds ========"

for SEED in "${SEEDS[@]}"; do
  # UniSRec Base
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] UniSRec Base (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model UniSRec \
    --dataset Amazon_Beauty \
    --config_files "unisrec_beauty_stratified_ts.yaml" \
    --config_dict "{'freeze_backbone': False}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions 5 \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "./saved/ts_unisrec_base_seed${SEED}" \
    --seed "$SEED" \
    --variant_features "unisrec,base,beauty,ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_unisrec_base_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"

  # UniSRec + AlignV3
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] UniSRec AlignV3 (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model UniSRecAlignV3 \
    --dataset Amazon_Beauty \
    --config_files "unisrec_align_v3_beauty_stratified_ts.yaml" \
    --config_dict "{'freeze_backbone': False}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions 5 \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "./saved/ts_unisrec_alignv3_seed${SEED}" \
    --seed "$SEED" \
    --variant_features "unisrec,alignv3,beauty,ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_unisrec_alignv3_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"

  # UniSRec + AlignMultiViewV3
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] UniSRec MV-V3 (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model UniSRecAlignMultiViewV3 \
    --dataset Amazon_Beauty \
    --config_files "unisrec_align_multiview_v3_beauty_stratified_ts.yaml" \
    --config_dict "{'freeze_backbone': False, 'use_text_view_senet': false}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions 5 \
    --phase_a_epochs 50 \
    --phase_a_eval_step 5 \
    --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "./saved/ts_unisrec_mv_v3_seed${SEED}" \
    --seed "$SEED" \
    --variant_features "unisrec,mv_v3,no_se,beauty,ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_unisrec_mv_v3_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"
done

log "======== GTS UniSRec complete ($RUN/$TOTAL runs) ========"
