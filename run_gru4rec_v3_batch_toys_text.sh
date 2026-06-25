#!/usr/bin/env bash
# GRU4Rec Toys: TF-IDF+LLM + MV only (ID/TF-IDF on log10).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
METRIC_BASELINE="${METRIC_BASELINE:-0.0272}"
export GPU_ID SEED METRIC_BASELINE

echo "1/2 toys GRU4Rec TF-IDF + LLM V3"
bash "$ROOT/two_phase_run_gru4rec_tfidf_llm_v3_toys_stratified.sh"
echo "2/2 toys GRU4Rec Multi-View V3"
bash "$ROOT/two_phase_run_gru4rec_multiview_v3_toys_stratified.sh"
