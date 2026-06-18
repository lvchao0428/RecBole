#!/usr/bin/env bash
# Pure SASRec baseline (50ep, V3) — book-crossing, multi-seed
# 单 seed: run50epBase_v3_book_crossing_stratified.sh
#
# 用法（项目根目录）:
#   bash run50epBase_v3_book_crossing_stratified_multiseed.sh
# 可选: SEEDS="42 2024 2025 2026"  GPU_ID=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${GPU_ID:-0}
SEEDS="${SEEDS:-42 2024 2025 2026}"

echo "========================================="
echo "SASRec Baseline (50ep, V3) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: SASRecAlignMultiViewV3 (text disabled = pure ID)"
echo "GPU: $GPU_ID"
echo "Seeds: $SEEDS"
echo ""

for SEED in $SEEDS; do
    echo "========================================="
    echo "Running seed: $SEED"
    echo "========================================="
    env SEED="$SEED" GPU_ID="$GPU_ID" bash "$ROOT/run50epBase_v3_book_crossing_stratified.sh"
    echo ""
    echo "Seed $SEED done."
    echo ""
done

echo "========================================="
echo "All seeds completed!"
echo "========================================="
