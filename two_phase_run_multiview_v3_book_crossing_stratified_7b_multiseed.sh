#!/usr/bin/env bash
# Multi-View V3 — book-crossing，多 seed 串行
# 单 seed：two_phase_run_multiview_v3_book_crossing_stratified_7b.sh
#
# 用法（项目根目录）:
#   bash two_phase_run_multiview_v3_book_crossing_stratified_7b_multiseed.sh
# 可选: SEEDS="42 2024 2025 2026"  GPU_ID=0  METRIC_BASELINE=0.015

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-42 2024 2025 2026}"

echo "========================================="
echo "Multi-View V3 (book-crossing) — multi-seed"
echo "Seeds: $SEEDS"
echo "========================================="
echo ""

for SEED in $SEEDS; do
  echo "========================================="
  echo "Running seed: $SEED"
  echo "========================================="
  env SEED="$SEED" bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"
  echo ""
  echo "✅ Seed $SEED done."
  echo ""
done

echo "========================================="
echo "✅ All seeds completed!"
echo "========================================="
