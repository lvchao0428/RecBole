#!/usr/bin/env bash
# Mechanism Analysis Queue (0723 zhidao requirements)
# 1. View redundancy (CKA + effective rank) — Beauty + Toys
# 2. Leave-one-view-out — Beauty (using LOO MV ckpt seed42)
# 3. Distribution shift / temporal analysis — formal output
#
# All serial, GPU only needed for leave-one-view-out.
#
# Usage:
#   nohup bash run_5090_mechanism_analysis.sh > logs/mechanism_analysis.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh" 2>/dev/null || true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOGFILE="logs/mechanism_analysis_queue.log"
mkdir -p logs paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "======== MECHANISM ANALYSIS START ========"

# ─── 1. View Redundancy (CPU only, ~2min each) ───
log ">>> [1/5] View Redundancy — Beauty"
python scripts/analyze_view_redundancy.py \
  --dataset Amazon_Beauty \
  --output paper_recsys/view_redundancy_beauty.json \
  2>&1 | tee logs/view_redundancy_beauty.log
log "  Beauty done"

log ">>> [2/5] View Redundancy — Toys"
python scripts/analyze_view_redundancy.py \
  --dataset Amazon_Toys_and_Games \
  --output paper_recsys/view_redundancy_toys.json \
  2>&1 | tee logs/view_redundancy_toys.log
log "  Toys done"

# ─── 2. Leave-One-View-Out (GPU, ~15min) ───
log ">>> [3/5] Leave-One-View-Out — Beauty (LOO MV seed42 ckpt)"
python scripts/analyze_leave_one_view_out.py \
  --ckpt saved/loo_beauty_mv_lr5e4_do0.3_seed42 \
  --dataset Amazon_Beauty \
  --output paper_recsys/leave_one_view_out_beauty_loo.md \
  2>&1 | tee logs/leave_one_view_out_beauty.log
log "  Beauty LOO done"

log ">>> [4/5] Leave-One-View-Out — Toys (LOO MV seed42 ckpt)"
python scripts/analyze_leave_one_view_out.py \
  --ckpt saved/loo_toys_mv_lr5e4_do0.3_seed2026 \
  --dataset Amazon_Toys_and_Games \
  --output paper_recsys/leave_one_view_out_toys_loo.md \
  2>&1 | tee logs/leave_one_view_out_toys.log
log "  Toys LOO done"

# ─── 3. Distribution Shift Analysis (CPU only, ~2min) ───
log ">>> [5/5] Distribution Shift — Beauty + Toys"
python scripts/analyze_distribution_shift.py \
  --dataset Amazon_Beauty \
  --output paper_recsys/distribution_shift_beauty_formal.json \
  2>&1 | tee logs/distribution_shift_beauty.log

python scripts/analyze_distribution_shift.py \
  --dataset Amazon_Toys_and_Games \
  --output paper_recsys/distribution_shift_toys_formal.json \
  2>&1 | tee logs/distribution_shift_toys.log
log "  Distribution shift done"

log "======== MECHANISM ANALYSIS COMPLETE ========"
log "Output files:"
log "  paper_recsys/view_redundancy_beauty.json"
log "  paper_recsys/view_redundancy_toys.json"
log "  paper_recsys/leave_one_view_out_beauty_loo.md"
log "  paper_recsys/leave_one_view_out_toys_loo.md"
log "  paper_recsys/distribution_shift_beauty_formal.json"
log "  paper_recsys/distribution_shift_toys_formal.json"
