#!/usr/bin/env bash
# 5090 单卡 ~24h 串行实验计划（2026-06-19）
#
# 前置：P0 趋势验证 seed2025 的 baseline / TF-IDF / single-view 已在 6/18 完成；
#       multi-view 因 views.json 缺失失败，本脚本从修复 + 补跑 multi-view 开始。
#
# 预估总耗时 ~22–24h（5090 单卡，book-crossing）:
#   Step 0  repair views.json              ~1 min
#   Step 1  SASRec multi-view seed2025      ~3.5 h
#   Step 2  SASRec baseline ×3 seeds       ~2.6 h  (42 / 2024 / 2026)
#   Step 3  GRU4Rec ×4 配置 seed2025       ~8–9 h
#   Step 4  FDSA ×4 配置 seed2025          ~8–9 h
#   Step 5  UniSRec seed2025              ~1.5 h
#
# 用法（RecBole 根目录，5090）:
#   nohup bash run_5090_24h_book_crossing_serial.sh \
#     > logs/exp_5090_24h_book_crossing_serial.log 2>&1 &
#
# 可选:
#   GPU_ID=0  SEED=2025  METRIC_BASELINE=0.028
#   SKIP_MULTIVIEW=1       # multi-view 已手动补完
#   SKIP_BASELINE_SEEDS=1  # baseline 三 seed 已齐
#   SKIP_GRU4REC=1  SKIP_FDSA=1  SKIP_UNISREC=1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
METRIC_BASELINE="${METRIC_BASELINE:-0.028}"
BASELINE_SEEDS="${BASELINE_SEEDS:-42 2024 2026}"

SKIP_MULTIVIEW="${SKIP_MULTIVIEW:-0}"
SKIP_BASELINE_SEEDS="${SKIP_BASELINE_SEEDS:-0}"
SKIP_GRU4REC="${SKIP_GRU4REC:-0}"
SKIP_FDSA="${SKIP_FDSA:-0}"
SKIP_UNISREC="${SKIP_UNISREC:-0}"

export GPU_ID SEED METRIC_BASELINE

LOG_DIR="${LOG_DIR:-$ROOT/logs/5090_24h_serial}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/00_master.log"

log_step() {
  local msg="$1"
  echo "" | tee -a "$MASTER_LOG"
  echo "========================================================================" | tee -a "$MASTER_LOG"
  echo ">>> $msg" | tee -a "$MASTER_LOG"
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
  echo "========================================================================" | tee -a "$MASTER_LOG"
}

run_substep() {
  local name="$1"
  shift
  local step_log="$LOG_DIR/${name}.log"
  log_step "$name"
  {
    echo "=== START: $(date '+%Y-%m-%d %H:%M:%S') ==="
    "$@"
    echo "=== END:   $(date '+%Y-%m-%d %H:%M:%S') ==="
  } 2>&1 | tee "$step_log"
}

echo "########################################################################" | tee "$MASTER_LOG"
echo "# 5090 ~24h Book-Crossing Serial Pipeline" | tee -a "$MASTER_LOG"
echo "# Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "# GPU=$GPU_ID  SEED=$SEED  METRIC_BASELINE=$METRIC_BASELINE" | tee -a "$MASTER_LOG"
echo "# Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "########################################################################" | tee -a "$MASTER_LOG"

# --- Step 0: repair views.json (multi-view blocker) ---
log_step "0/5 repair views.json (if missing)"
VIEWS_JSON="$ROOT/dataset/book-crossing/qwen3_4views/views.json"
if [[ ! -f "$VIEWS_JSON" ]]; then
  python "$ROOT/tools/repair_qwen3_views_json.py" \
    --split_dir "$ROOT/dataset/book-crossing/qwen3_4views" | tee -a "$MASTER_LOG"
else
  echo "views.json exists — skip repair" | tee -a "$MASTER_LOG"
fi

# --- Step 1: P0 补跑 SASRec multi-view seed2025 ---
if [[ "$SKIP_MULTIVIEW" != "1" ]]; then
  run_substep "01_sasrec_multiview_seed${SEED}" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh"
else
  log_step "1/5 SASRec multi-view — SKIP"
fi

# --- Step 2: SASRec V3 baseline 补 3 seeds ---
if [[ "$SKIP_BASELINE_SEEDS" != "1" ]]; then
  for s in $BASELINE_SEEDS; do
    run_substep "02_sasrec_baseline_seed${s}" \
      env SEED="$s" GPU_ID="$GPU_ID" \
      bash "$ROOT/run50epBase_v3_book_crossing_stratified.sh"
  done
else
  log_step "2/5 SASRec baseline multiseed — SKIP"
fi

# --- Step 3: GRU4Rec ×4 配置 ---
if [[ "$SKIP_GRU4REC" != "1" ]]; then
  run_substep "03_gru4rec_batch_seed${SEED}" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/run_gru4rec_v3_batch_book_crossing.sh"
else
  log_step "3/5 GRU4Rec batch — SKIP"
fi

# --- Step 4: FDSA ×4 配置 ---
if [[ "$SKIP_FDSA" != "1" ]]; then
  run_substep "04_fdsa_batch_seed${SEED}" \
    env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/run_fdsa_v3_batch_book_crossing.sh"
else
  log_step "4/5 FDSA batch — SKIP"
fi

# --- Step 5: UniSRec ---
if [[ "$SKIP_UNISREC" != "1" ]]; then
  run_substep "05_unisrec_seed${SEED}" \
    env SEED="$SEED" GPU_ID="$GPU_ID" \
    bash "$ROOT/run_unisrec_book_crossing_stratified.sh"
else
  log_step "5/5 UniSRec — SKIP"
fi

echo "" | tee -a "$MASTER_LOG"
echo "########################################################################" | tee -a "$MASTER_LOG"
echo "# ALL DONE: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "# Per-step logs: $LOG_DIR/" | tee -a "$MASTER_LOG"
echo "########################################################################" | tee -a "$MASTER_LOG"
