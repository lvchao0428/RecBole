#!/usr/bin/env bash
# Two-phase TF-IDF + Qwen3 LLM V3 — book-crossing，多 seed 串行
# 单 seed：two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh
#
# 用法（项目根目录）:
#   bash two_phase_run_tfidf_llm_v3_book_crossing_stratified_multiseed.sh
# 可选: SEEDS="42 2024 2025 2026"  GPU_ID=0  METRIC_BASELINE=0.015

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-42 2024 2025 2026}"

echo "========================================="
echo "TF-IDF + LLM V3 (book-crossing) — multi-seed"
echo "Seeds: $SEEDS"
echo "========================================="
echo ""

for SEED in $SEEDS; do
  echo "========================================="
  echo "Running seed: $SEED"
  echo "========================================="
  env SEED="$SEED" bash "$ROOT/two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh"
  echo ""
  echo "✅ Seed $SEED done."
  echo ""
done

echo "========================================="
echo "✅ All seeds completed!"
echo "========================================="
