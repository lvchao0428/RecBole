#!/usr/bin/env bash
# =============================================================================
# Master Batch Script: All Backbone & Baseline Experiments
#
# Runs the full experiment matrix:
#   - GRU4Rec x 4 configs x 3 datasets
#   - FDSA x 4 configs x 3 datasets
#   - UniSRec x 3 datasets
#
# Usage:
#   GPU_ID=0 SEED=2025 bash run_all_backbone_baseline_batch.sh
#   GPU_ID=0 bash run_all_backbone_baseline_batch.sh   # defaults: SEED=2025
#
# Multi-seed usage:
#   for SEED in 2025 42 123 2024; do
#     GPU_ID=0 SEED=$SEED bash run_all_backbone_baseline_batch.sh
#   done
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
export GPU_ID SEED

echo "============================================================"
echo "  Master Batch: All Backbone & Baseline Experiments"
echo "  GPU=$GPU_ID  SEED=$SEED"
echo "  Start: $(date)"
echo "============================================================"

# ---- GRU4Rec (4 configs x 3 datasets = 12 runs) ----
echo ""
echo "============ GRU4Rec: Beauty ============"
bash "$ROOT/run_gru4rec_v3_batch_beauty.sh"

echo ""
echo "============ GRU4Rec: Toys ============"
bash "$ROOT/run_gru4rec_v3_batch_toys.sh"

echo ""
echo "============ GRU4Rec: book-crossing ============"
bash "$ROOT/run_gru4rec_v3_batch_book_crossing.sh"

# ---- FDSA (4 configs x 3 datasets = 12 runs) ----
echo ""
echo "============ FDSA: Beauty ============"
bash "$ROOT/run_fdsa_v3_batch_beauty.sh"

echo ""
echo "============ FDSA: Toys ============"
bash "$ROOT/run_fdsa_v3_batch_toys.sh"

echo ""
echo "============ FDSA: book-crossing ============"
bash "$ROOT/run_fdsa_v3_batch_book_crossing.sh"

# ---- UniSRec (3 datasets = 3 runs) ----
echo ""
echo "============ UniSRec: All datasets ============"
bash "$ROOT/run_unisrec_batch_all.sh"

echo ""
echo "============================================================"
echo "  Master Batch DONE"
echo "  End: $(date)"
echo "============================================================"
