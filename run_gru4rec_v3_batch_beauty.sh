#!/usr/bin/env bash
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

echo "1/4 beauty GRU4Rec ID-only Baseline"
bash "$ROOT/run50epBase_gru4rec_v3_beauty_stratified.sh"
echo "2/4 beauty GRU4Rec TF-IDF V3"
bash "$ROOT/two_phase_run_gru4rec_tfidf_v3_beauty_stratified.sh"
echo "3/4 beauty GRU4Rec TF-IDF + LLM V3"
bash "$ROOT/two_phase_run_gru4rec_tfidf_llm_v3_beauty_stratified.sh"
echo "4/4 beauty GRU4Rec Multi-View V3"
bash "$ROOT/two_phase_run_gru4rec_multiview_v3_beauty_stratified.sh"
