#!/usr/bin/env bash
# 串行执行 book-crossing 四套 Stratified 实验：上一段成功结束后才跑下一段；任一步失败则立即退出。
# 每一步 stdout/stderr 写入单独日志文件，并同时打印到终端（tee）。
#
# 用法（在 RecBole 仓库根目录）:
#   bash run_all_book_crossing_stratified_sequential.sh
# 可选环境变量:
#   GPU_ID=0 METRIC_BASELINE=0.015
#   LOG_DIR=/path/to/dir          # 指定目录（否则见下）
#   RUN_ID=20260101_120000        # 与 LOG_DIR 二选一：默认自动生成时间戳子目录
#   DATASET_BOOK_CROSSING_DIR=... # book-crossing 数据目录，默认 <ROOT>/dataset/book-crossing

#set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET_BOOK_CROSSING_DIR="${DATASET_BOOK_CROSSING_DIR:-$ROOT/dataset/book-crossing}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/book_crossing_stratified/$RUN_ID}"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/00_pipeline_master.log"

log_master() {
  echo "$@" | tee -a "$MASTER_LOG"
}

log_master "########################################################################"
log_master "# Book-Crossing Stratified 串行流水线"
log_master "# 顺序: 预处理 .inter 补 timestamp → 60ep baseline → TF-IDF V3 → TF-IDF+LLM V3 → Multi-View V3"
log_master "# 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 日志目录: $LOG_DIR"
log_master "# 子日志: 00_preprocess_timestamp.log, 01…04 …"
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

run_step "0/5 book-crossing.inter 补 timestamp（已有则跳过）" "$LOG_DIR/00_preprocess_timestamp.log" \
  python "$ROOT/tools/preprocess_book_crossing_inter_add_timestamp.py" \
    --dataset_dir "$DATASET_BOOK_CROSSING_DIR" \
    --in_place \
    --backup

run_step "1/5 SASRec baseline (60ep)" "$LOG_DIR/01_baseline_60ep.log" \
  bash "$ROOT/run60epBase_book_crossing_stratified.sh"

run_step "2/5 Two-phase TF-IDF V3" "$LOG_DIR/02_tfidf_v3.log" \
  bash "$ROOT/two_phase_run_tfidf_v3_book_crossing_stratified.sh"

run_step "3/5 Two-phase TF-IDF + Qwen3 LLM V3" "$LOG_DIR/03_tfidf_llm_v3.log" \
  bash "$ROOT/two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh"

run_step "4/5 Two-phase Multi-View V3 (qwen3_4views)" "$LOG_DIR/04_multiview_v3.log" \
  bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"

log_master "########################################################################"
log_master "# 全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 汇总索引: $MASTER_LOG"
log_master "########################################################################"
