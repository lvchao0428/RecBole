#!/usr/bin/env bash
# Phase 5: UniSRec +Align / +MV balanced (Beauty seed=2024, infer_boost=0.6)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "UniSRec balanced batch (seed=$SEED, GPU=$GPU_ID)"

env GPU_ID="$GPU_ID" SEED="$SEED" \
  bash "$ROOT/run_unisrec_align_v3_beauty_stratified_balanced.sh" \
  >> "$LOG_DIR/unisrec_align_v3_beauty_balanced_seed${SEED}.log" 2>&1

env GPU_ID="$GPU_ID" SEED="$SEED" \
  bash "$ROOT/run_unisrec_align_multiview_v3_beauty_stratified_balanced.sh" \
  >> "$LOG_DIR/unisrec_align_multiview_v3_beauty_balanced_seed${SEED}.log" 2>&1

echo "UniSRec balanced batch complete."
