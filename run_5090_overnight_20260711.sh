#!/usr/bin/env bash
# 5090 Overnight Pipeline — 2026-07-11 21:00 start
# 
# Queue:
#   1. ID-only min5 baseline (Beauty, seed=2025)                 ~26min
#   2. TF-IDF + frequency weight boost (Beauty, seed=2025)       ~78min
#   3. TF-IDF+LLM + frequency weight boost (Beauty, seed=2025)  ~101min
#   4. MV-Align + frequency weight boost (Beauty, seed=2025)     ~204min
#   5. Toys/Grocery TS-aware Qwen embedding regeneration         ~10min
#   6. Beauty 3-seed (TF-IDF no-boost, seed=2024,42)             ~156min
#
# Total est: ~9.5h → finish ~06:30
#
# Usage:
#   nohup bash run_5090_overnight_20260711.sh > logs/overnight_20260711_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
PHASE_A_EPOCHS=20
PHASE_B_EPOCHS=50
MIN_TRAIN=5
TFIDF_TS="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
QWEN_TS="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
VIEWS_TS_DIR="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"
LOG="logs/overnight_20260711.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== 5090 Overnight Pipeline Start ========"

###############################################################################
# 1. ID-only min5 baseline (Beauty, seed=2025)
###############################################################################
log ">>> [1/6] ID-only min5 baseline (Beauty, seed=2025)"
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
  --checkpoint_dir ./saved/ts_beauty_id_only_min5_seed2025 \
  --seed 2025 \
  --variant_features "sasrec,id_only,beauty,ts,min5,seed2025" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_id_only_min5_seed2025.log" 2>&1
log "  done ($((SECONDS - t0))s)"

###############################################################################
# 2. TF-IDF + frequency weight boost (Beauty, seed=2025)
###############################################################################
log ">>> [2/6] TF-IDF + freq-weight boost (Beauty, seed=2025)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.0, 'cold_threshold': 10, 'item_text_emb_path_base': '${TFIDF_TS}'}" \
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
  --checkpoint_dir ./saved/ts_beauty_tfidf_boost_seed2025 \
  --seed 2025 \
  --variant_features "sasrec,tfidf_ts,v3,beauty,ts,cold_boost3,min5,seed2025" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_tfidf_boost_seed2025.log" 2>&1
log "  done ($((SECONDS - t0))s)"

###############################################################################
# 3. TF-IDF+LLM + frequency weight boost (Beauty, seed=2025)
###############################################################################
log ">>> [3/6] TF-IDF+LLM + freq-weight boost (Beauty, seed=2025)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.0, 'cold_threshold': 10, 'item_text_emb_path_base': '${TFIDF_TS}', 'item_text_emb_path_llm': '${QWEN_TS}'}" \
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
  --checkpoint_dir ./saved/ts_beauty_llm_boost_seed2025 \
  --seed 2025 \
  --variant_features "sasrec,tfidf_ts,llm,v3,beauty,ts,cold_boost3,min5,seed2025" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_llm_boost_seed2025.log" 2>&1
log "  done ($((SECONDS - t0))s)"

###############################################################################
# 4. MV-Align + frequency weight boost (Beauty, seed=2025)
###############################################################################
log ">>> [4/6] MV-Align + freq-weight boost (Beauty, seed=2025)"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.0, 'cold_threshold': 10, 'use_text_view_senet': false, 'item_text_emb_path_base': '${TFIDF_TS}', 'item_text_emb_split_dir': '${VIEWS_TS_DIR}'}" \
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
  --checkpoint_dir ./saved/ts_beauty_mv_boost_seed2025 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v3,7b,4views,beauty,ts,cold_boost3,no_se,min5,seed2025" \
  --watchdog_disable \
  --save \
  >> "logs/ts_beauty_mv_boost_seed2025.log" 2>&1
log "  done ($((SECONDS - t0))s)"

###############################################################################
# 5. Generate Toys/Grocery TS-aware Qwen embeddings
###############################################################################
log ">>> [5/6] Generating Toys/Grocery TS-aware Qwen embeddings"
t0=$SECONDS

for DS in Amazon_Toys_and_Games Amazon_Grocery_and_Gourmet_Food; do
  DS_DIR="$ROOT/dataset/$DS"
  if [ "$DS" = "Amazon_Toys_and_Games" ]; then
    CFG="sasrec_align_toys_base_stratified_v3_ts.yaml"
  else
    CFG="sasrec_align_grocery_base_stratified_v3_ts.yaml"
  fi
  
  log "  Rebuilding Qwen embeddings for $DS ..."
  python tools/rebuild_qwen_emb_ts.py \
    --dataset_dir "$DS_DIR" \
    --dataset "$DS" \
    --config "$CFG" \
    2>&1 | tee -a "$LOG"
done
log "  Qwen embedding regeneration done ($((SECONDS - t0))s)"

###############################################################################
# 6. Beauty 3-seed expansion: TF-IDF no-boost (seed=2024, 42)
###############################################################################
log ">>> [6/6] Beauty 3-seed: TF-IDF no-boost (seed=2024, 42)"
for SEED in 2024 42; do
  log "  TF-IDF no-boost seed=$SEED"
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
done

log "======== 5090 Overnight Pipeline Complete ========"
log "Expected finish: ~06:30 next morning"
