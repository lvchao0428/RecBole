#!/usr/bin/env bash
# Regenerate ALL item text features with strict TS-aware leakage prevention
#
# For each dataset (Beauty, Toys, Grocery):
#   1. TF-IDF: vocabulary/IDF + SVD fit on train only; center+whiten on train only
#   2. Single-view Qwen: center+whiten on train only (no re-inference needed)
#   3. Multi-view Qwen: invert old whiten → re-apply center+whiten on train only
#
# After regeneration, launches the full experiment queue.
#
# Usage:
#   nohup bash run_5090_regen_all_ts_v3.sh > logs/regen_all_ts_v3_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG="logs/regen_all_ts_v3.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== Regenerating ALL TS-aware features (V3, with center+whiten) ========"

###############################################################################
# Beauty
###############################################################################
log "=== [1/3] Amazon_Beauty ==="

# TF-IDF with center+whiten
log "  TF-IDF (center+whiten) ..."
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_baseline_50ep_stratified_ts.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  --center --whiten \
  2>&1 | tee -a "$LOG"
log "  TF-IDF done ($((SECONDS - t0))s)"

# Single-view Qwen (center+whiten with TS train_ids)
log "  Single-view Qwen + Multi-view Qwen ..."
t0=$SECONDS
python tools/rebuild_qwen_emb_ts.py \
  --dataset Amazon_Beauty \
  --config sasrec_baseline_50ep_stratified_ts.yaml \
  2>&1 | tee -a "$LOG"
log "  Qwen done ($((SECONDS - t0))s)"

###############################################################################
# Toys
###############################################################################
log "=== [2/3] Amazon_Toys_and_Games ==="

log "  TF-IDF (center+whiten) ..."
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_baseline_50ep_stratified_toys_ts.yaml \
  --output dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  --center --whiten \
  2>&1 | tee -a "$LOG"
log "  TF-IDF done ($((SECONDS - t0))s)"

log "  Single-view Qwen + Multi-view Qwen ..."
t0=$SECONDS
python tools/rebuild_qwen_emb_ts.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_baseline_50ep_stratified_toys_ts.yaml \
  2>&1 | tee -a "$LOG"
log "  Qwen done ($((SECONDS - t0))s)"

###############################################################################
# Grocery
###############################################################################
log "=== [3/3] Amazon_Grocery_and_Gourmet_Food ==="

log "  TF-IDF (center+whiten) ..."
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Grocery_and_Gourmet_Food \
  --config sasrec_baseline_50ep_stratified_grocery_ts.yaml \
  --output dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  --center --whiten \
  2>&1 | tee -a "$LOG"
log "  TF-IDF done ($((SECONDS - t0))s)"

log "  Single-view Qwen + Multi-view Qwen ..."
t0=$SECONDS
python tools/rebuild_qwen_emb_ts.py \
  --dataset Amazon_Grocery_and_Gourmet_Food \
  --config sasrec_baseline_50ep_stratified_grocery_ts.yaml \
  2>&1 | tee -a "$LOG"
log "  Qwen done ($((SECONDS - t0))s)"

###############################################################################
# Verify outputs
###############################################################################
log "======== Verification ========"
for DS in Amazon_Beauty Amazon_Toys_and_Games Amazon_Grocery_and_Gourmet_Food; do
  log "--- $DS ---"
  ls -lh "dataset/$DS/item_text_emb.base.ts.npy" \
         "dataset/$DS/item_text_emb.base.ts_whiten_stats.npz" \
         "dataset/$DS/item_text_emb.qwen2.5_7b.base.ts.npy" \
    2>&1 | tee -a "$LOG"
  ls -lh "dataset/$DS/qwen2.5_7b_4views_ts/"*.npy \
    2>&1 | tee -a "$LOG"
done

log "======== ALL features regenerated ========"

###############################################################################
# Now launch the experiment queue
###############################################################################
log ">>> Launching experiment queue (run_5090_queue_20260712.sh) ..."
bash run_5090_queue_20260712.sh 2>&1 | tee -a "$LOG"

log "======== Pipeline complete ========"
