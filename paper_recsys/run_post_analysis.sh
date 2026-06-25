#!/usr/bin/env bash
# Zero-GPU post-training analysis bundle (0617/0618 supplemental metrics).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG="${POST_ANALYSIS_LOG:-logs/post_analysis_20260625.log}"
mkdir -p logs paper_recsys ablation_study_doc/figures

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log ">>> Post-analysis Phase A: tables & significance (0 GPU)"

log "  [1/4] compute_table_stats.py → main_table.txt"
python paper_recsys/compute_table_stats.py >> "$LOG" 2>&1

log "  [2/4] extract_coverage.py"
python paper_recsys/extract_coverage.py >> "$LOG" 2>&1

log "  [3/4] per-user paired t-test (Beauty + Toys)"
{
  echo "=== Beauty TF-IDF+LLM vs MV-Align ==="
  python paper_recsys/compute_peruser_significance.py \
    --model_a saved/peruser/beauty_tfidf_llm_topk.npy \
    --model_b saved/peruser/beauty_mv_7b_topk.npy \
    --label_a "TF-IDF+LLM" --label_b "MV-Align(7B)"
  echo ""
  echo "=== Toys TF-IDF+LLM vs MV-Align ==="
  python paper_recsys/compute_peruser_significance.py \
    --model_a saved/peruser/toys_tfidf_llm_topk.npy \
    --model_b saved/peruser/toys_mv_7b_topk.npy \
    --label_a "TF-IDF+LLM" --label_b "MV-Align(7B)"
} | tee paper_recsys/peruser_significance_20260625.txt >> "$LOG"

log "  [4/4] MV gate bucket analysis"
python paper_recsys/analyze_mv_gate_buckets.py >> "$LOG" 2>&1

log ">>> Post-analysis Phase A complete"
