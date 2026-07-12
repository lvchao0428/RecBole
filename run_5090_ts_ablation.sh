#!/usr/bin/env bash
# GTS Ablation — Beauty × 3 ablations × 1 seed on 5090
# Ablations: no-Whiten, no-Cross, no-Align
#
# Usage (on 5090):
#   nohup bash run_5090_ts_ablation.sh \
#     > logs/ts_ablation_5090_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
LOG="logs/ts_ablation_5090.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS Ablation: Beauty × 3 ablations (seed=$SEED) ========"

# 1. No Whitening
log ">>> [1/3] MV-Align no-Whiten"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_beauty_ts_no_whiten.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_mv_no_whiten_seed${SEED}" \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,no_whiten,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_no_whiten_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 2. No Cross
log ">>> [2/3] MV-Align no-Cross"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_mv_nocross_seed${SEED}" \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,nocross,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_nocross_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 3. No Align
log ">>> [3/3] MV-Align no-Align"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_beauty_ts_no_align.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/ts_beauty_mv_no_align_seed${SEED}" \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,no_align,beauty,ts,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_no_align_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

log "======== GTS Ablation complete ========"
