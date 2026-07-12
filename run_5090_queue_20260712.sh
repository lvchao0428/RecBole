#!/usr/bin/env bash
# 5090 Experiment Queue — 2026-07-12
# Based on advisor 0712 guidance: verify 3-seed stability, Cross ablation, Toys/Grocery
#
# Phase 1 (排查) — SKIP: V2 已完成且 V1 异常已解释 (freeze_backbone bug)
# Phase 2: Beauty 3-seed (LLM x2 + MV x2)              ~610 min / ~10h
# Phase 3: Cross ablation (MV no-Cross)                 ~200 min / ~3.3h
# Phase 4: Toys/Grocery 4-config each                   ~814 min / ~13.5h
#
# Total est: ~27h → start 12 Jul 13:30 → finish ~14 Jul 16:30
#
# Usage:
#   nohup bash run_5090_queue_20260712.sh > logs/queue_20260712_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
PA_EP=20
PB_EP=50
MIN5=5

# TS-aware feature paths
B_TFIDF="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts.npy"
B_QWEN="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
B_VIEWS="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts"

T_TFIDF="$ROOT/dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy"
T_QWEN="$ROOT/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy"
T_VIEWS="$ROOT/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_ts"

G_TFIDF="$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy"
G_QWEN="$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.qwen2.5_7b.base.ts.npy"
G_VIEWS="$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food/qwen2.5_7b_4views_ts"

LOG="logs/queue_20260712.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_text_model() {
  local MODEL="$1" CONFIG="$2" LABEL="$3" LOGFILE="$4" CKPT_DIR="$5" SEED="$6"
  shift 6
  local EXTRA_DICT="$*"
  
  log "  >>> $LABEL (seed=$SEED)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset Amazon_Beauty \
    --config_files "$CONFIG" \
    --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0${EXTRA_DICT:+, $EXTRA_DICT}}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_grid \
    --align_grid "0.10" \
    --tau_grid "0.05" \
    --backbone_burnin_epochs 0 \
    --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP \
    --phase_a_eval_step 1 \
    --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 \
    --metric_gain_threshold 0.01 \
    --lr_text_head 2e-3 \
    --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b \
    --phase_b_epochs $PB_EP \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --variant_features "ts,noboost,min5,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

run_id_only() {
  local DATASET="$1" CONFIG="$2" LABEL="$3" LOGFILE="$4" CKPT_DIR="$5" SEED="$6"
  
  log "  >>> $LABEL (seed=$SEED)"
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
    --checkpoint_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --variant_features "id_only,ts,min5,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

run_text_general() {
  local MODEL="$1" DATASET="$2" CONFIG="$3" LABEL="$4" LOGFILE="$5" CKPT_DIR="$6" SEED="$7"
  shift 7
  local EXTRA_DICT="$*"
  
  log "  >>> $LABEL (seed=$SEED)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --config_files "$CONFIG" \
    --config_dict "{'align_weight': 0.1, 'cold_text_boost': 0.0, 'infer_boost': 0.0, 'cold_threshold': 0${EXTRA_DICT:+, $EXTRA_DICT}}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_grid \
    --align_grid "0.10" \
    --tau_grid "0.05" \
    --backbone_burnin_epochs 0 \
    --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP \
    --phase_a_eval_step 1 \
    --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 \
    --metric_gain_threshold 0.01 \
    --lr_text_head 2e-3 \
    --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b \
    --phase_b_epochs $PB_EP \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --variant_features "ts,noboost,min5,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "$LOGFILE" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

log "======== 5090 Queue 0712 Start ========"

###############################################################################
# Phase 2: Beauty 3-seed expansion (LLM + MV, seed=2024/42)
###############################################################################
log "=== Phase 2: Beauty 3-seed (LLM x2 + MV x2) ==="

for SEED in 2024 42; do
  run_text_model SASRecAlignV3 \
    "sasrec_align_qwen3_stratified_v3_ts.yaml" \
    "Beauty TF-IDF+LLM no-boost" \
    "logs/ts_beauty_llm_v2_seed${SEED}.log" \
    "./saved/ts_beauty_llm_v2_seed${SEED}" \
    "$SEED" \
    "'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_path_llm': '${B_QWEN}'"
done

for SEED in 2024 42; do
  run_text_model SASRecAlignMultiViewV3 \
    "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    "Beauty MV-Align no-boost" \
    "logs/ts_beauty_mv_v2_seed${SEED}.log" \
    "./saved/ts_beauty_mv_v2_seed${SEED}" \
    "$SEED" \
    "'use_text_view_senet': false, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}'"
done

log "=== Phase 2 complete ==="

###############################################################################
# Phase 3: Cross ablation — MV no-Cross (Beauty, seed=2025)
###############################################################################
log "=== Phase 3: Cross ablation ==="

run_text_model SASRecAlignMultiViewV3 \
  "sasrec_align_multi_view_v3_beauty_ts_nocross.yaml" \
  "Beauty MV-Align NO-CROSS" \
  "logs/ts_beauty_mv_nocross_v2_seed2025.log" \
  "./saved/ts_beauty_mv_nocross_v2_seed2025" \
  2025 \
  "'use_text_view_senet': false, 'use_cross': false, 'use_multiview_text_cross': false, 'item_text_emb_path_base': '${B_TFIDF}', 'item_text_emb_split_dir': '${B_VIEWS}'"

log "=== Phase 3 complete ==="

###############################################################################
# Phase 4: Toys 4-config (seed=2025, min5, no-boost)
###############################################################################
log "=== Phase 4a: Toys 4-config ==="

# Toys ID-only
run_id_only Amazon_Toys_and_Games \
  "sasrec_baseline_50ep_stratified_toys_ts.yaml" \
  "Toys ID-only min5" \
  "logs/ts_toys_id_only_min5_seed2025.log" \
  "./saved/ts_toys_id_only_min5_seed2025" \
  2025

# Toys TF-IDF
run_text_general SASRecAlignV3 Amazon_Toys_and_Games \
  "sasrec_align_toys_base_stratified_v3_ts.yaml" \
  "Toys TF-IDF no-boost" \
  "logs/ts_toys_tfidf_v2_seed2025.log" \
  "./saved/ts_toys_tfidf_v2_seed2025" \
  2025 \
  "'item_text_emb_path_base': '${T_TFIDF}'"

# Toys TF-IDF+LLM
run_text_general SASRecAlignV3 Amazon_Toys_and_Games \
  "sasrec_align_toys_qwen3_stratified_v3_ts.yaml" \
  "Toys TF-IDF+LLM no-boost" \
  "logs/ts_toys_llm_v2_seed2025.log" \
  "./saved/ts_toys_llm_v2_seed2025" \
  2025 \
  "'item_text_emb_path_base': '${T_TFIDF}', 'item_text_emb_path_llm': '${T_QWEN}'"

# Toys MV-Align
run_text_general SASRecAlignMultiViewV3 Amazon_Toys_and_Games \
  "sasrec_align_multi_view_v3_toys_stratified_7b_ts.yaml" \
  "Toys MV-Align no-boost" \
  "logs/ts_toys_mv_v2_seed2025.log" \
  "./saved/ts_toys_mv_v2_seed2025" \
  2025 \
  "'use_text_view_senet': false, 'item_text_emb_path_base': '${T_TFIDF}', 'item_text_emb_split_dir': '${T_VIEWS}'"

log "=== Phase 4a: Toys complete ==="

###############################################################################
# Phase 4b: Grocery 4-config (seed=2025, min5, no-boost)
###############################################################################
log "=== Phase 4b: Grocery 4-config ==="

# Grocery ID-only
run_id_only Amazon_Grocery_and_Gourmet_Food \
  "sasrec_baseline_50ep_stratified_grocery_ts.yaml" \
  "Grocery ID-only min5" \
  "logs/ts_grocery_id_only_min5_seed2025.log" \
  "./saved/ts_grocery_id_only_min5_seed2025" \
  2025

# Grocery TF-IDF
run_text_general SASRecAlignV3 Amazon_Grocery_and_Gourmet_Food \
  "sasrec_align_grocery_base_stratified_v3_ts.yaml" \
  "Grocery TF-IDF no-boost" \
  "logs/ts_grocery_tfidf_v2_seed2025.log" \
  "./saved/ts_grocery_tfidf_v2_seed2025" \
  2025 \
  "'item_text_emb_path_base': '${G_TFIDF}'"

# Grocery TF-IDF+LLM
run_text_general SASRecAlignV3 Amazon_Grocery_and_Gourmet_Food \
  "sasrec_align_grocery_qwen_stratified_v3_ts.yaml" \
  "Grocery TF-IDF+LLM no-boost" \
  "logs/ts_grocery_llm_v2_seed2025.log" \
  "./saved/ts_grocery_llm_v2_seed2025" \
  2025 \
  "'item_text_emb_path_base': '${G_TFIDF}', 'item_text_emb_path_llm': '${G_QWEN}'"

# Grocery MV-Align
run_text_general SASRecAlignMultiViewV3 Amazon_Grocery_and_Gourmet_Food \
  "sasrec_align_multi_view_v3_grocery_stratified_7b_ts.yaml" \
  "Grocery MV-Align no-boost" \
  "logs/ts_grocery_mv_v2_seed2025.log" \
  "./saved/ts_grocery_mv_v2_seed2025" \
  2025 \
  "'use_text_view_senet': false, 'item_text_emb_path_base': '${G_TFIDF}', 'item_text_emb_split_dir': '${G_VIEWS}'"

log "=== Phase 4b: Grocery complete ==="

log "======== 5090 Queue 0712 Complete ========"
