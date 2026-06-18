#!/usr/bin/env bash
# P0 最高优先级：book-crossing 单 seed 四配置趋势验证
# 目的：对比 Beauty/Toys，确认 baseline → tfidf → single-view → multi-view 主指标走势一致
#       不一致则先调 balanced 超参，再 multiseed / backbone
#
# 用法（RecBole 根目录）:
#   nohup bash run_book_crossing_v3_trend_check.sh \
#     > logs/exp_bc_v3_trend_check_seed2025.log 2>&1 &
#
# 可选:
#   SEED=2025  GPU_ID=0
#   SKIP_BASELINE=1          # baseline 已有则跳过（默认 1）
#   METRIC_BASELINE=0.028    # 默认取 seed2025 V3 baseline valid MRR@10

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
# V3 baseline seed2025 valid MRR@10 ≈ 0.0278 (2026-06-18)
METRIC_BASELINE="${METRIC_BASELINE:-0.028}"

export GPU_ID SEED METRIC_BASELINE

echo "########################################################################"
echo "# Book-Crossing V3 趋势验证（单 seed 四配置）"
echo "# SEED=$SEED  GPU=$GPU_ID  METRIC_BASELINE=$METRIC_BASELINE"
echo "# SKIP_BASELINE=$SKIP_BASELINE"
echo "# Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"

if [[ "$SKIP_BASELINE" != "1" ]]; then
  echo ""
  echo ">>> 1/4 baseline (ID-only, 50ep V3)"
  env SEED="$SEED" GPU_ID="$GPU_ID" bash "$ROOT/run50epBase_v3_book_crossing_stratified.sh"
else
  echo ""
  echo ">>> 1/4 baseline — SKIP (use saved/baseline_v3_book_crossing_stratified_seed${SEED})"
fi

echo ""
echo ">>> 2/4 TF-IDF V3"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_tfidf_v3_book_crossing_stratified.sh"

echo ""
echo ">>> 3/4 single-view (TF-IDF + Qwen3 LLM) V3"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh"

echo ""
echo ">>> 4/4 multi-view (Qwen3 4-view) V3"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"

echo ""
echo "########################################################################"
echo "# 趋势验证跑完: $(date '+%Y-%m-%d %H:%M:%S')"
echo "# 对比 Beauty/Toys 同配置 test MRR@10 / Recall@10 / stratified new-item"
echo "# Checkpoints:"
echo "#   saved/baseline_v3_book_crossing_stratified_seed${SEED}"
echo "#   saved/two_phase_run_tfidf_v3_book_crossing_stratified_seed${SEED}"
echo "#   saved/two_phase_run_tfidf_llm_v3_book_crossing_stratified_seed${SEED}"
echo "#   saved/two_phase_run_multiview_v3_book_crossing_stratified_7b_seed${SEED}"
echo "########################################################################"
