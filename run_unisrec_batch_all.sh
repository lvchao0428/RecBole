#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
export GPU_ID SEED

echo "========================================="
echo "UniSRec Batch: all 3 datasets"
echo "GPU=$GPU_ID  SEED=$SEED"
echo "========================================="

echo "1/3 UniSRec - Beauty"
bash "$ROOT/run_unisrec_beauty_stratified.sh"

echo "2/3 UniSRec - Toys"
bash "$ROOT/run_unisrec_toys_stratified.sh"

echo "3/3 UniSRec - book-crossing"
bash "$ROOT/run_unisrec_book_crossing_stratified.sh"

echo "========================================="
echo "UniSRec Batch DONE"
echo "========================================="
