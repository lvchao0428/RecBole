#!/usr/bin/env bash
# 串行跑完 book-crossing 剩余实验（2026-06-18 续跑）
#
# === Book-Crossing 完整矩阵（4 配置 × 多 backbone）===
# 配置命名:
#   baseline     = ID-only
#   tfidf        = SASRecAlignV3 + TF-IDF
#   single-view  = TF-IDF + Qwen3 LLM（脚本名 tfidf_llm）
#   multi-view   = Qwen3 4-view MV-Align
#
# | Backbone | baseline | tfidf | single-view | multi-view |
# |----------|----------|-------|-------------|------------|
# | SASRec   | 补 3 seeds | ✅ 4 seeds (Apr) | ✅ 4 seeds | ✅ 4 seeds |
# | GRU4Rec  | 本脚本 step 2 | 本脚本 step 2 | 本脚本 step 2 | 本脚本 step 2 |
# | FDSA     | 本脚本 step 3 | 本脚本 step 3 | 本脚本 step 3 | 本脚本 step 3 |
# | UniSRec  | — | LLM only (step 4) | — | — |
#
# 本脚本顺序:
#   1) SASRec V3 baseline 补 seed 42 / 2024 / 2026（text 三配置 Apr 已齐，不重跑）
#   2) GRU4Rec × 4 配置 (baseline / tfidf / single-view / multi-view), seed=2025
#   3) FDSA × 4 配置, seed=2025
#   4) UniSRec, seed=2025
#
# 若需 SASRec 四配置 × 四 seed 从头重跑:
#   bash run_0125Batch_v3_book_crossing_multiseed.sh
#
# 用法（RecBole 根目录）:
#   nohup bash run_all_book_crossing_remaining_serial.sh \
#     > logs/exp_20260618_bc_remaining_serial.log 2>&1 &
#
# 可选: GPU_ID=0  SEED=2025  BASELINE_SEEDS="42 2024 2026"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
BASELINE_SEEDS="${BASELINE_SEEDS:-42 2024 2026}"

export GPU_ID SEED METRIC_BASELINE

log_step() {
  echo ""
  echo "========================================================================"
  echo ">>> $1"
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================================================"
}

echo "########################################################################"
echo "# Book-Crossing 剩余实验串行流水线"
echo "# 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "# GPU_ID=$GPU_ID  SEED=$SEED  BASELINE_SEEDS=$BASELINE_SEEDS"
echo "#"
echo "# SASRec tfidf / single-view / multi-view: 5090 已有 4 seeds (Apr)，跳过"
echo "# 5090 saved/: two_phase_run_tfidf_v3_book_crossing_stratified_seed*"
echo "#              two_phase_run_tfidf_llm_v3_book_crossing_stratified_seed*"
echo "#              two_phase_run_multiview_v3_book_crossing_stratified_7b_seed*"
echo "########################################################################"

log_step "1/4 SASRec V3 baseline 补 seed: $BASELINE_SEEDS"
for s in $BASELINE_SEEDS; do
  echo "--- baseline V3 seed=$s ---"
  env SEED="$s" GPU_ID="$GPU_ID" bash "$ROOT/run50epBase_v3_book_crossing_stratified.sh"
done

log_step "2/4 GRU4Rec V3: baseline → tfidf → single-view → multi-view (seed=$SEED)"
bash "$ROOT/run_gru4rec_v3_batch_book_crossing.sh"

log_step "3/4 FDSA V3: baseline → tfidf → single-view → multi-view (seed=$SEED)"
bash "$ROOT/run_fdsa_v3_batch_book_crossing.sh"

log_step "4/4 UniSRec book-crossing (seed=$SEED)"
bash "$ROOT/run_unisrec_book_crossing_stratified.sh"

echo ""
echo "########################################################################"
echo "# 全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"
