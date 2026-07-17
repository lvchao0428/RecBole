#!/usr/bin/env bash
# GTS Multi-seed Beauty — Text models on 5090 (seeds 2024, 42)
# seed=2025 already done in pilot; reuse those results.
# Runs: 3 text configs × 2 seeds = 6 runs
#
# Usage (on 5090):
#   nohup bash run_5090_ts_beauty_3seed.sh \
#     > logs/ts_beauty_3seed_5090_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
LOG="logs/ts_beauty_3seed_5090.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

SEEDS=(2024 42)

log "======== GTS Beauty 3-seed: Text models on 5090 ========"
log "Seeds to run: ${SEEDS[*]}  (seed=2025 already done)"
log "Configs: TF-IDF, TF-IDF+LLM, MV-Align (no SE)"

RUN=0
TOTAL=$((3 * ${#SEEDS[@]}))

for SEED in "${SEEDS[@]}"; do
  log "-------- Seed=$SEED --------"

  # TF-IDF
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] TF-IDF (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlignV3 \
    --dataset Amazon_Beauty \
    --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
    --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
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
    --checkpoint_dir ./saved/ts_beauty_tfidf_seed${SEED} \
    --seed "$SEED" \
    --variant_features "sasrec,tfidf,v3,beauty,ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_beauty_tfidf_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"

  # TF-IDF + LLM
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] TF-IDF+LLM (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlignV3 \
    --dataset Amazon_Beauty \
    --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
    --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
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
    --checkpoint_dir ./saved/ts_beauty_tfidf_llm_seed${SEED} \
    --seed "$SEED" \
    --variant_features "sasrec,tfidf,llm,v3,beauty,ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_beauty_tfidf_llm_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"

  # MV-Align (SE disabled)
  RUN=$((RUN + 1))
  log ">>> [$RUN/$TOTAL] MV-Align no-SE (seed=$SEED)"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlignMultiViewV3 \
    --dataset Amazon_Beauty \
    --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
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
    --checkpoint_dir ./saved/ts_beauty_mv_seed${SEED} \
    --seed "$SEED" \
    --variant_features "sasrec,multiview_v3,7b,4views,beauty,ts,no_se,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_beauty_mv_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"
done

log "======== GTS Beauty 3-seed (5090 text) complete ========"
log "Completed $RUN/$TOTAL runs"
