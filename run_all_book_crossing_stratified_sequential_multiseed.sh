#!/usr/bin/env bash
# 串行执行 book-crossing Stratified 流水线（多 Seed 版本）
# 对应 run_all_book_crossing_stratified_sequential.sh 中的五步：
#   0) 预处理  1) 60ep baseline  2) TF-IDF V3  3) TF-IDF+LLM V3  4) Multi-View V3
# 每一步内按 SEEDS 串行跑完所有 seed 再进入下一步。
# 每个模型 × 每个 seed 产生独立日志文件，方便查看。
#
# 默认 SEEDS="42 2024 2025 2026"（论文 4 个 seed）
#
# 用法（RecBole 仓库根目录）:
#   bash run_all_book_crossing_stratified_sequential_multiseed.sh
# 可选环境变量:
#   GPU_ID=0  METRIC_BASELINE=0.015
#   SEEDS="42 2024 2025 2026"
#   LOG_DIR=...  RUN_ID=...
#   DATASET_BOOK_CROSSING_DIR=...

#set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
SEEDS="${SEEDS:-42 2024 2025 2026}"

DATASET_BOOK_CROSSING_DIR="${DATASET_BOOK_CROSSING_DIR:-$ROOT/dataset/book-crossing}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/book_crossing_stratified_multiseed/$RUN_ID}"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/00_pipeline_master.log"

log_master() {
  echo "$@" | tee -a "$MASTER_LOG"
}

log_master "########################################################################"
log_master "# Book-Crossing Stratified 串行流水线（多 Seed）"
log_master "# 顺序: 预处理 → baseline(×|SEEDS|) → TF-IDF V3(×|SEEDS|) → TF-IDF+LLM V3(×|SEEDS|) → Multi-View V3(×|SEEDS|)"
log_master "# SEEDS: $SEEDS"
log_master "# 开始: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 日志目录: $LOG_DIR"
log_master "# GPU_ID=$GPU_ID  METRIC_BASELINE=$METRIC_BASELINE"
log_master "########################################################################"
log_master ""

run_step() {
  local step_label="$1"
  local log_file="$2"
  shift 2

  log_master ""
  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  log_master ">>> STEP: $step_label"
  log_master ">>> 子日志: $log_file"
  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  {
    echo "=== $step_label ==="
    echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    "$@"
    echo ""
    echo "结束: $(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee "$log_file"

  log_master ""
  log_master ">>> 完成: $step_label  ($(date '+%Y-%m-%d %H:%M:%S'))"
}

# ---------- 0: 预处理（只跑一次） ----------
run_step "0/5 book-crossing.inter 补 timestamp（已有则跳过）" "$LOG_DIR/00_preprocess_timestamp.log" \
  python "$ROOT/tools/preprocess_book_crossing_inter_add_timestamp.py" \
    --dataset_dir "$DATASET_BOOK_CROSSING_DIR" \
    --in_place \
    --backup

# ---------- 1: SASRec baseline 60ep × seeds ----------
for SEED in $SEEDS; do
  run_step "1/5 SASRec baseline (60ep)  seed=${SEED}" "$LOG_DIR/01_baseline_60ep_seed${SEED}.log" \
    env SEED="$SEED" GPU_ID="$GPU_ID" \
    bash "$ROOT/run60epBase_book_crossing_stratified.sh"
done

# ---------- 2: Two-phase TF-IDF V3 × seeds ----------
for SEED in $SEEDS; do
  run_step "2/5 Two-phase TF-IDF V3  seed=${SEED}" "$LOG_DIR/02_tfidf_v3_seed${SEED}.log" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/two_phase_run_tfidf_v3_book_crossing_stratified.sh"
done

# ---------- 3: Two-phase TF-IDF + Qwen3 LLM V3 × seeds ----------
for SEED in $SEEDS; do
  run_step "3/5 Two-phase TF-IDF + Qwen3 LLM V3  seed=${SEED}" "$LOG_DIR/03_tfidf_llm_v3_seed${SEED}.log" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh"
done

# ---------- 4: Multi-View V3 × seeds ----------
for SEED in $SEEDS; do
  run_step "4/5 Two-phase Multi-View V3 (qwen3_4views)  seed=${SEED}" "$LOG_DIR/04_multiview_v3_seed${SEED}.log" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"
done

log_master "########################################################################"
log_master "# 全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 汇总索引: $MASTER_LOG"
log_master "########################################################################"
