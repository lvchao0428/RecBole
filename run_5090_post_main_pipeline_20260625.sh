#!/usr/bin/env bash
# Phase 3: mechanism analysis + supplemental metrics (after main training pipeline).
#
# Queued automatically at the end of run_5090_main_pipeline_20260625*.sh
#
#   3A  0 GPU  — main_table, Coverage, per-user t-test, gate buckets
#   3B  GPU    — Beauty save_test_scores (ID/TF-IDF/MV, eval-only)
#   3C  0 GPU  — concentration plots + case-study summary
#   4   GPU    — GRU4Rec V3 Beauty+Toys × 4 seeds × 4 configs (second backbone)

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

log "======== Post Main Pipeline 20260625 (Phase 3) ========"
log "GPU=$GPU_ID"

# --- 3A: zero-GPU analysis ---
log ">>> Phase 3A: supplemental tables & significance"
t0=$SECONDS
bash "$ROOT/paper_recsys/run_post_analysis.sh" \
  >> "$LOG" 2>&1
log "  ✅ Phase 3A done ($((SECONDS - t0))s)"

# --- 3B: GPU eval-only scores (0617 mechanism #1 & #3) ---
log ">>> Phase 3B: Beauty save_test_scores (ID/TF-IDF/MV)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 \
  bash "$ROOT/run_beauty_mechanism_save_scores.sh" \
  >> "$LOG" 2>&1
log "  ✅ Phase 3B done ($((SECONDS - t0))s)"

# --- 3C: plots from saved scores ---
log ">>> Phase 3C: mechanism figures"
SCORES="$ROOT/ablation_study_doc/scores"
FIGS="$ROOT/ablation_study_doc/figures"
mkdir -p "$FIGS"

ID_SCORES="$SCORES/beauty_id_only_seed2024_mechanism_phase_b_topk_scores.npy"
TFIDF_SCORES="$SCORES/beauty_tfidf_v3_seed2024_mechanism_phase_b_topk_scores.npy"
MV_SCORES="$SCORES/beauty_mv_v3_seed2024_mechanism_phase_b_topk_scores.npy"

if [[ -f "$ID_SCORES" && -f "$TFIDF_SCORES" ]]; then
  log "  plot: ID vs TF-IDF concentration"
  python ablation_study_doc/visualize_ablation.py \
    --plot-score-dist \
    --baseline-scores "$ID_SCORES" \
    --ablation-scores "$TFIDF_SCORES" \
    --baseline-label "ID-only" \
    --ablation-label "TF-IDF" \
    --output-dir "$FIGS/id_vs_tfidf" >> "$LOG" 2>&1 || log "  ⚠️ id_vs_tfidf plot failed"
fi

if [[ -f "$TFIDF_SCORES" && -f "$MV_SCORES" ]]; then
  log "  plot: TF-IDF vs MV concentration"
  python ablation_study_doc/visualize_ablation.py \
    --plot-score-dist \
    --baseline-scores "$TFIDF_SCORES" \
    --ablation-scores "$MV_SCORES" \
    --baseline-label "TF-IDF" \
    --ablation-label "MV-Align" \
    --output-dir "$FIGS/tfidf_vs_mv" >> "$LOG" 2>&1 || log "  ⚠️ tfidf_vs_mv plot failed"
fi

# Case study: summarize per-user topk shifts (LLM vs MV already on disk)
log "  case study: LLM vs MV topk summary"
{
  echo "Case study summary (existing per-user topk, Beauty seed2025 reruns)"
  echo "Files: saved/peruser/beauty_tfidf_llm_topk.npy vs beauty_mv_7b_topk.npy"
  echo "Mechanism topk (this run): $PERUSER_DIR or saved/peruser/mechanism/"
  ls -la saved/peruser/mechanism/*_topk.npy 2>/dev/null || true
} | tee paper_recsys/mechanism_case_study_notes_20260625.txt >> "$LOG"

# --- Phase 4: GRU4Rec second backbone (4 seeds × Beauty/Toys × 4 configs) ---
log ">>> Phase 4: GRU4Rec V3 multiseed (Beauty + Toys)"
t0=$SECONDS
env GPU_ID="$GPU_ID" USE_LOG10="${USE_LOG10:-1}" \
  bash "$ROOT/run_gru4rec_v3_multiseed_beauty_toys.sh" \
  >> "$LOG" 2>&1
log "  ✅ Phase 4 GRU4Rec done ($((SECONDS - t0))s)"

log "======== Post Main Pipeline 20260625 complete ========"
