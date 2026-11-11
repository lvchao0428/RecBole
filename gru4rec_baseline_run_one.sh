#!/usr/bin/env bash
# Shared GRU4Rec ID + TF-IDF block (Beauty or Toys, one seed).
# Used by 5090 and log10; skips steps when checkpoint already exists.
#
# Env:
#   DATASET=beauty|toys   SEED=2024   GPU_ID=0
#   METRIC_BASELINE=0.0272
#   TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE (log10 uses 256)
#   SKIP_ID=1|0  SKIP_TFIDF=1|0
#   MACHINE=5090|log10  (log prefix only)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:?DATASET required (beauty|toys)}"
SEED="${SEED:?SEED required}"
GPU_ID="${GPU_ID:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.0272}"
SKIP_ID="${SKIP_ID:-0}"
SKIP_TFIDF="${SKIP_TFIDF:-0}"
MACHINE="${MACHINE:-5090}"

export GPU_ID SEED METRIC_BASELINE
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

has_ckpt() {
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "${dir}/*.pth" >/dev/null
}

id_ckpt="./saved/gru4rec_baseline_v3_${DATASET}_stratified_seed${SEED}"
tfidf_ckpt="./saved/gru4rec_tfidf_v3_${DATASET}_stratified_seed${SEED}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MACHINE] $*"; }

log "--- GRU4Rec ID+TF-IDF ${DATASET} seed=${SEED} batch=${TRAIN_BATCH_SIZE} ---"
t0=$SECONDS

if [[ "$DATASET" == "beauty" ]]; then
  id_script="$ROOT/run50epBase_gru4rec_v3_beauty_stratified.sh"
  tfidf_script="$ROOT/two_phase_run_gru4rec_tfidf_v3_beauty_stratified.sh"
  id_log="logs/${MACHINE}_gru4rec_beauty_id_seed${SEED}.log"
  tfidf_log="logs/${MACHINE}_gru4rec_beauty_tfidf_seed${SEED}.log"
else
  id_script="$ROOT/run50epBase_gru4rec_v3_toys_stratified.sh"
  tfidf_script="$ROOT/two_phase_run_gru4rec_tfidf_v3_toys_stratified.sh"
  id_log="logs/${MACHINE}_gru4rec_toys_id_seed${SEED}.log"
  tfidf_log="logs/${MACHINE}_gru4rec_toys_tfidf_seed${SEED}.log"
fi

if [[ "$SKIP_ID" != "1" ]] && has_ckpt "$id_ckpt"; then
  log "  skip ID (checkpoint exists): $id_ckpt"
  SKIP_ID=1
fi
if [[ "$SKIP_TFIDF" != "1" ]] && has_ckpt "$tfidf_ckpt"; then
  log "  skip TF-IDF (checkpoint exists): $tfidf_ckpt"
  SKIP_TFIDF=1
fi

if [[ "$SKIP_ID" != "1" ]]; then
  env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$id_script" >> "$id_log" 2>&1
fi
if [[ "$SKIP_TFIDF" != "1" ]]; then
  env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$tfidf_script" >> "$tfidf_log" 2>&1
fi

log "  ✅ ${DATASET} seed=${SEED} done ($((SECONDS - t0))s)"
