#!/usr/bin/env bash
# Resume post pipeline from Phase 3B (3A already done 2026-06-25 21:54).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID="${GPU_ID:-0}"
LOG="logs/post_main_pipeline_20260625.log"
mkdir -p logs ablation_study_doc/figures paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Post Pipeline RESUME from 3B ========"

log ">>> Phase 3B: Beauty save_test_scores (eval batch=64, no peruser)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 TRAIN_BATCH_SIZE=256 EVAL_BATCH_SIZE=64 SAVE_PERUSER=0 \
  bash "$ROOT/run_beauty_mechanism_save_scores.sh" >> "$LOG" 2>&1
log "  ✅ Phase 3B done ($((SECONDS - t0))s)"

log ">>> Phase 3C: mechanism figures"
SCORES="$ROOT/ablation_study_doc/scores"
FIGS="$ROOT/ablation_study_doc/figures"
mkdir -p "$FIGS"
ID_SCORES="$SCORES/beauty_id_only_seed2024_mechanism_phase_b_topk_scores.npy"
TFIDF_SCORES="$SCORES/beauty_tfidf_v3_seed2024_mechanism_phase_b_topk_scores.npy"
MV_SCORES="$SCORES/beauty_mv_v3_seed2024_mechanism_phase_b_topk_scores.npy"

if [[ -f "$ID_SCORES" && -f "$TFIDF_SCORES" ]]; then
  python ablation_study_doc/visualize_ablation.py --plot-score-dist \
    --baseline-scores "$ID_SCORES" --ablation-scores "$TFIDF_SCORES" \
    --baseline-label "ID-only" --ablation-label "TF-IDF" \
    --output-dir "$FIGS/id_vs_tfidf" >> "$LOG" 2>&1 || true
fi
if [[ -f "$TFIDF_SCORES" && -f "$MV_SCORES" ]]; then
  python ablation_study_doc/visualize_ablation.py --plot-score-dist \
    --baseline-scores "$TFIDF_SCORES" --ablation-scores "$MV_SCORES" \
    --baseline-label "TF-IDF" --ablation-label "MV-Align" \
    --output-dir "$FIGS/tfidf_vs_mv" >> "$LOG" 2>&1 || true
fi

log ">>> Phase 4: GRU4Rec (5090 text + log10 baselines)"
t0=$SECONDS
env GPU_ID="$GPU_ID" USE_LOG10=1 \
  bash "$ROOT/run_gru4rec_v3_multiseed_beauty_toys.sh" >> "$LOG" 2>&1
log "  ✅ Phase 4 done ($((SECONDS - t0))s)"

log "======== Post Pipeline RESUME complete ========"
