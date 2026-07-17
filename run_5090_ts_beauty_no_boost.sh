#!/usr/bin/env bash
# GTS Beauty — 去掉 infer_boost 和 cold_text_boost 的简化版
# 3 个 text 配置 × 1 seed (2025), 串行跑在 5090
#
# Usage (on 5090):
#   nohup bash run_5090_ts_beauty_no_boost.sh \
#     > logs/ts_beauty_no_boost_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED=2025
PHASE_A_EPOCHS=20
PHASE_B_EPOCHS=50
LOG="logs/ts_beauty_no_boost.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS Beauty NO-BOOST: 3 text configs (seed=$SEED) ========"
log "cold_text_boost=0, infer_boost=0, cold_threshold=0"

# 1. TF-IDF
log ">>> [1/3] TF-IDF (SASRecAlignV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0}" \
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
  --checkpoint_dir ./saved/ts_beauty_tfidf_noboost_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf,v3,beauty,ts,noboost,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_noboost_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 2. TF-IDF + LLM
log ">>> [2/3] TF-IDF+LLM (SASRecAlignV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0}" \
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
  --checkpoint_dir ./saved/ts_beauty_tfidf_llm_noboost_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf,llm,v3,beauty,ts,noboost,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_llm_noboost_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 3. MV-Align (SE disabled)
log ">>> [3/3] MV-Align no-SE (SASRecAlignMultiViewV3)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'use_text_view_senet': false}" \
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
  --checkpoint_dir ./saved/ts_beauty_mv_noboost_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,7b,4views,beauty,ts,noboost,no_se,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_noboost_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

log "======== GTS Beauty NO-BOOST complete ========"
