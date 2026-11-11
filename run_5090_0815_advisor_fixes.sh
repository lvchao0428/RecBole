#!/usr/bin/env bash
set -euo pipefail
source scripts/recbole_env.sh
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
LOG=logs/advisor_fixes_0815.log
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

SEEDS=(42 2024 2026)

# ============================================================
# PART 1: Equal-Horizon Early [0.6,0.1,0.1] for Beauty
#   TF / LLM / MV × 3 seeds
#   Fixes RQ2 confound where Early used [0.6,0.1,0.3]
# ============================================================

log "=== PART 1: Equal-Horizon Early60-EQ on Beauty ==="

for SEED in "${SEEDS[@]}"; do
  log "--- TF Early60-EQ Beauty seed=$SEED ---"
  python scripts/two_phase_train.py \
    -m SASRecAlignV3 -d Amazon_Beauty \
    --config_files "sasrec_align_base_stratified_v3_ts.yaml" \
    --config_dict "{'eval_args': {'group_by': 'user', 'order': 'TO', 'split': {'TS': [0.6, 0.1, 0.1]}, 'mode': {'valid': 'full', 'test': 'full'}}}" \
    --seed "$SEED" --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "TF_early60eq_s${SEED}" \
    2>&1 | tee -a "$LOG"

  log "--- LLM Early60-EQ Beauty seed=$SEED ---"
  python scripts/two_phase_train.py \
    -m SASRecAlignV3 -d Amazon_Beauty \
    --config_files "sasrec_align_qwen3_stratified_v3_ts.yaml" \
    --config_dict "{'eval_args': {'group_by': 'user', 'order': 'TO', 'split': {'TS': [0.6, 0.1, 0.1]}, 'mode': {'valid': 'full', 'test': 'full'}}}" \
    --seed "$SEED" --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "LLM_early60eq_s${SEED}" \
    2>&1 | tee -a "$LOG"

  log "--- MV Early60-EQ Beauty seed=$SEED ---"
  python scripts/two_phase_train.py \
    -m SASRecAlignMultiViewV3 -d Amazon_Beauty \
    --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    --config_dict "{'eval_args': {'group_by': 'user', 'order': 'TO', 'split': {'TS': [0.6, 0.1, 0.1]}, 'mode': {'valid': 'full', 'test': 'full'}}}" \
    --seed "$SEED" --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "MV_early60eq_s${SEED}" \
    2>&1 | tee -a "$LOG"
done

log "=== PART 1 DONE ==="

# ============================================================
# PART 2: TF+LLM Hybrid — Beauty and Toys × 3 seeds
#   MV architecture with text_view_indices=[0] → TF base + 1 LLM view
#   Tests if TF+LLM complementarity (CKA~0.19) yields additive gain
# ============================================================

log "=== PART 2: TF+LLM Hybrid ==="

for SEED in "${SEEDS[@]}"; do
  log "--- TF+LLM Hybrid Beauty seed=$SEED ---"
  python scripts/two_phase_train.py \
    -m SASRecAlignMultiViewV3 -d Amazon_Beauty \
    --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    --config_dict "{'num_text_views': 1, 'text_view_indices': [0]}" \
    --seed "$SEED" --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "TF_LLM_hybrid_beauty_s${SEED}" \
    2>&1 | tee -a "$LOG"

  log "--- TF+LLM Hybrid Toys seed=$SEED ---"
  python scripts/two_phase_train.py \
    -m SASRecAlignMultiViewV3 -d Amazon_Toys_and_Games \
    --config_files "sasrec_align_multi_view_v3_stratified_ts.yaml" \
    --config_dict "{'dataset': 'Amazon_Toys_and_Games', 'num_text_views': 1, 'text_view_indices': [0], 'item_text_emb_path_base': '/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy', 'item_text_emb_split_dir': '/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views'}" \
    --seed "$SEED" --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "TF_LLM_hybrid_toys_s${SEED}" \
    2>&1 | tee -a "$LOG"
done

log "=== PART 2 DONE ==="

log "=== ALL ADVISOR FIX EXPERIMENTS COMPLETE ==="
touch logs/advisor_fixes_0815.DONE
