#!/usr/bin/env bash
# book-crossing MV：显式 Phase-B=50ep 实验（主表协议）
# Phase-A 20ep = grid/warmup（不计入可比训练量）；Phase-B 50ep 与 ID-only 50ep 对齐
#
# Usage:
#   METRIC_BASELINE=0.028 SEED=2025 GPU_ID=0 \
#     bash two_phase_run_multiview_v3_book_crossing_stratified_7b_phaseb50.sh
#
# 旧 phaseb40 结果保留在 saved/*_seed${SEED}/（无 phaseb50 后缀）

set -euo pipefail

export PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
export PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
export CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"
