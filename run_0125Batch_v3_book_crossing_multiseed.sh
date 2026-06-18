#!/usr/bin/env bash
# V3 Batch Training Script — Book-Crossing (multi-seed)
# 对应 run_0125Batch_v3.sh 的 book-crossing 多 seed 版本
# 每个 seed 串行跑完全部 4 个配置 (baseline → TF-IDF → LLM → Multi-View)
#
# 用法（项目根目录）:
#   bash run_0125Batch_v3_book_crossing_multiseed.sh
# 可选:
#   GPU_ID=0  SEEDS="42 2024 2025 2026"  METRIC_BASELINE=0.015
#   LOG_DIR=...  RUN_ID=...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-42 2024 2025 2026}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/book_crossing_v3_batch/$RUN_ID}"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/00_master.log"

log_master() {
  echo "$@" | tee -a "$MASTER_LOG"
}

log_master "########################################################################"
log_master "# Book-Crossing V3 Batch (multi-seed)"
log_master "# Seeds: $SEEDS"
log_master "# GPU_ID=$GPU_ID  METRIC_BASELINE=$METRIC_BASELINE"
log_master "# Start: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# Log dir: $LOG_DIR"
log_master "########################################################################"
log_master ""

TOTAL_SEEDS=$(echo $SEEDS | wc -w | tr -d ' ')
SEED_IDX=0

for SEED in $SEEDS; do
  SEED_IDX=$((SEED_IDX + 1))
  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  log_master ">>> Seed $SEED_IDX/$TOTAL_SEEDS: $SEED"
  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  STEP_LOG="$LOG_DIR/seed_${SEED}.log"

  {
    echo "=== seed=$SEED  start: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/run_0125Batch_v3_book_crossing.sh"
    echo ""
    echo "=== seed=$SEED  end: $(date '+%Y-%m-%d %H:%M:%S') ==="
  } 2>&1 | tee "$STEP_LOG"

  log_master ">>> Seed $SEED done ($(date '+%Y-%m-%d %H:%M:%S'))"
  log_master ""
done

log_master "########################################################################"
log_master "# All done: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# Master log: $MASTER_LOG"
log_master "########################################################################"
