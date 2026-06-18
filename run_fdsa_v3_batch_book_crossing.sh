#!/usr/bin/env bash
# FDSA V3 Batch Training Script — Book-Crossing
# Serial run: baseline → TF-IDF → TF-IDF+LLM → Multi-View
#
# Usage (from project root):
#   bash run_fdsa_v3_batch_book_crossing.sh
# Optional:
#   GPU_ID=0  SEED=2025  METRIC_BASELINE=0.015

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"

export GPU_ID SEED METRIC_BASELINE

echo "=============================================="
echo "FDSA V3 Batch Training — Book-Crossing"
echo "=============================================="
echo "Models:"
echo "  1) FDSAAlignMultiViewV3 (ID-only baseline)"
echo "  2) FDSAAlignV3          (TF-IDF)"
echo "  3) FDSAAlignV3          (TF-IDF + Qwen3 LLM)"
echo "  4) FDSAAlignMultiViewV3 (Multi-View)"
echo ""
echo "Balanced Params (via --config_dict):"
echo "  align_weight=0.1  cold_text_boost=3.0  infer_boost=0.6  cold_threshold=10"
echo ""
echo "GPU: $GPU_ID  SEED: $SEED  METRIC_BASELINE: $METRIC_BASELINE"
echo "=============================================="
echo ""

echo "1/4 [Book-Crossing] ID-only Baseline (50ep, V3)"
bash "$ROOT/run50epBase_fdsa_v3_book_crossing_stratified.sh"
echo "    Done!"
echo ""

echo "2/4 [Book-Crossing] TF-IDF FDSA V3"
bash "$ROOT/two_phase_run_fdsa_tfidf_v3_book_crossing_stratified.sh"
echo "    Done!"
echo ""

echo "3/4 [Book-Crossing] TF-IDF + Qwen3 LLM FDSA V3"
bash "$ROOT/two_phase_run_fdsa_tfidf_llm_v3_book_crossing_stratified.sh"
echo "    Done!"
echo ""

echo "4/4 [Book-Crossing] Multi-View FDSA V3"
bash "$ROOT/two_phase_run_fdsa_multiview_v3_book_crossing_stratified.sh"
echo "    Done!"
echo ""

echo "=============================================="
echo "All FDSA V3 Book-Crossing Training Tasks Done!"
echo "=============================================="
echo ""
echo "Check saved/ for checkpoints:"
echo "  - saved/baseline_fdsa_v3_book_crossing_stratified_seed${SEED}"
echo "  - saved/two_phase_run_fdsa_tfidf_v3_book_crossing_stratified_seed${SEED}"
echo "  - saved/two_phase_run_fdsa_tfidf_llm_v3_book_crossing_stratified_seed${SEED}"
echo "  - saved/two_phase_run_fdsa_multiview_v3_book_crossing_stratified_seed${SEED}"
echo ""
