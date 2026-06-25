#!/usr/bin/env bash
# UniSRec Beauty 三配置批量实验
#
# 三行主表实验：
#   1. UniSRec Base         (ID + MoE, no cross/align)
#   2. UniSRecAlignV3       (ID + MoE + Cross + Align)
#   3. UniSRecAlignMultiViewV3 (ID + MoE + Cross + Align + Multi-View)
#
# Usage:
#   SEED=2024 bash run_unisrec_v3_batch_beauty.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "UniSRec Beauty Batch (seed=$SEED, GPU=$GPU_ID)"
echo "========================================="

# --- Exp 1: UniSRec Base (original, additive only) ---
echo ""
echo "[1/3] UniSRec Base (ID + MoE, no cross/align)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED="$SEED" \
  bash "$ROOT/run_unisrec_beauty_stratified.sh" \
  >> "$LOG_DIR/unisrec_base_beauty_seed${SEED}.log" 2>&1
echo "  Done ($((SECONDS - t0))s)"

# --- Exp 2: UniSRecAlignV3 (+ Cross + Align) ---
echo ""
echo "[2/3] UniSRecAlignV3 (+ Cross + Align)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED="$SEED" \
  bash "$ROOT/run_unisrec_align_v3_beauty_stratified.sh" \
  >> "$LOG_DIR/unisrec_align_v3_beauty_seed${SEED}.log" 2>&1
echo "  Done ($((SECONDS - t0))s)"

# --- Exp 3: UniSRecAlignMultiViewV3 (+ Cross + Align + MV) ---
echo ""
echo "[3/3] UniSRecAlignMultiViewV3 (+ Cross + Align + MV)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED="$SEED" \
  bash "$ROOT/run_unisrec_align_multiview_v3_beauty_stratified.sh" \
  >> "$LOG_DIR/unisrec_align_multiview_v3_beauty_seed${SEED}.log" 2>&1
echo "  Done ($((SECONDS - t0))s)"

echo ""
echo "========================================="
echo "All 3 UniSRec Beauty experiments complete!"
echo "========================================="
