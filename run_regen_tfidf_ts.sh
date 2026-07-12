#!/usr/bin/env bash
# Regenerate TF-IDF embeddings with TS-aware train/test split
# TF-IDF vocabulary/IDF and SVD are fit ONLY on items appearing before train cutoff
# Output: item_text_emb.base.ts.npy for each dataset
#
# Usage (on 5090):
#   nohup bash run_regen_tfidf_ts.sh > logs/regen_tfidf_ts.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG="logs/regen_tfidf_ts.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== Regenerating TS-aware TF-IDF embeddings ========"

# Beauty
log ">>> [1/3] Amazon_Beauty"
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_baseline_50ep_stratified_ts.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
log "  Beauty done ($((SECONDS - t0))s)"

# Toys
log ">>> [2/3] Amazon_Toys_and_Games"
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_baseline_50ep_stratified_toys_ts.yaml \
  --output dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
log "  Toys done ($((SECONDS - t0))s)"

# Grocery
log ">>> [3/3] Amazon_Grocery_and_Gourmet_Food"
t0=$SECONDS
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Grocery_and_Gourmet_Food \
  --config sasrec_baseline_50ep_stratified_grocery_ts.yaml \
  --output dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy \
  --title_field title \
  --svd_dim 256 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
log "  Grocery done ($((SECONDS - t0))s)"

log "======== All TS-aware TF-IDF embeddings generated ========"
log "New files:"
ls -lh dataset/Amazon_Beauty/item_text_emb.base.ts.npy \
      dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy \
      dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy \
  2>&1 | tee -a "$LOG"
