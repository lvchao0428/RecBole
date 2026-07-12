#!/usr/bin/env bash
# GTS Beauty Pilot V2 — leakage-free TF-IDF + eligible user filtering
# 3 text configs × 1 seed (2025), no SE, no infer/cold boost
# Uses TS-aware TF-IDF (item_text_emb.base.ts.npy)
# Only evaluates users with >= 5 train interactions
#
# ID-only runs on log10 separately
#
# Usage (on 5090):
#   nohup bash run_5090_ts_beauty_pilot_v2.sh \
#     > logs/ts_beauty_pilot_v2_nohup.log 2>&1 &

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
MIN_TRAIN=5
TFIDF_TS="/home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
QWEN_TS="/home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
VIEWS_TS_DIR="/home/charlie/project/RecBole/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"
LOG="logs/ts_beauty_pilot_v2.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== GTS Beauty Pilot V2 (seed=$SEED) ========"
log "Features: TS-aware TF-IDF+Qwen (center/whiten on train only), min_train_interactions=$MIN_TRAIN, no boost, no SE"

# 1. TF-IDF (TS-aware)
log ">>> [1/3] TF-IDF (SASRecAlignV3) — TS-aware embedding"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'item_text_emb_path_base': '${TFIDF_TS}'}" \
  --gpu_id "$GPU_ID" \
  --min_train_interactions $MIN_TRAIN \
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
  --checkpoint_dir ./saved/ts_beauty_tfidf_v2_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf_ts,v3,beauty,ts,noboost,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_v2_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 2. TF-IDF (TS-aware) + LLM (TS-aware Qwen center/whiten)
log ">>> [2/3] TF-IDF+LLM (SASRecAlignV3) — TS-aware TF-IDF + TS-aware Qwen"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'item_text_emb_path_base': '${TFIDF_TS}', 'item_text_emb_path_llm': '${QWEN_TS}'}" \
  --gpu_id "$GPU_ID" \
  --min_train_interactions $MIN_TRAIN \
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
  --checkpoint_dir ./saved/ts_beauty_tfidf_llm_v2_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf_ts,llm,v3,beauty,ts,noboost,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_llm_v2_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

# 3. MV-Align (TS-aware TF-IDF + TS-aware multi-view Qwen, no SE)
log ">>> [3/3] MV-Align no-SE (SASRecAlignMultiViewV3) — TS-aware all features"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0, 'use_text_view_senet': false, 'item_text_emb_path_base': '${TFIDF_TS}', 'item_text_emb_split_dir': '${VIEWS_TS_DIR}'}" \
  --gpu_id "$GPU_ID" \
  --min_train_interactions $MIN_TRAIN \
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
  --checkpoint_dir ./saved/ts_beauty_mv_v2_seed${SEED} \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,7b,4views,beauty,ts,noboost,no_se,min5,seed${SEED}" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_v2_seed${SEED}.log" 2>&1
log "  done ($((SECONDS - t0))s)"

log "======== GTS Beauty Pilot V2 complete ========"
