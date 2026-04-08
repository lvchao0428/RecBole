#!/usr/bin/env bash
# 串行执行 book-crossing「Two-phase TF-IDF + Qwen3 LLM V3 (Stratified)」多 seed 实验。
# 与 two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh 参数一致，仅 seed 与 checkpoint 目录按 seed 区分。
#
# 默认 seeds：2005（当前常用）+ 论文常用 42, 2024, 2025, 2026（共 5 次）。
# 若论文只报 4 个 seed，可： SEEDS="42 2024 2025 2026" bash run_all_book_crossing_tfidf_llm_v3_seeds.sh
#
# 用法（RecBole 仓库根目录）:
#   bash run_all_book_crossing_tfidf_llm_v3_seeds.sh
# 可选环境变量:
#   GPU_ID=0 METRIC_BASELINE=0.015
#   SEEDS="2005 42 2024 2025 2026"
#   LOG_DIR=/path/to/dir
#   RUN_ID=20260101_120000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
# 默认五套；仅跑论文四 seed 时覆盖 SEEDS
SEEDS="${SEEDS:-2005 42 2024 2025 2026}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/book_crossing_tfidf_llm_v3_seeds/$RUN_ID}"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/00_master.log"

log_master() {
  echo "$@" | tee -a "$MASTER_LOG"
}

log_master "########################################################################"
log_master "# Book-Crossing TF-IDF + LLM V3 多 Seed 串行流水线"
log_master "# seeds: $SEEDS"
log_master "# 开始: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 日志目录: $LOG_DIR"
log_master "# GPU_ID=$GPU_ID  METRIC_BASELINE=$METRIC_BASELINE"
log_master "########################################################################"
log_master ""

run_one_seed() {
  local SEED="$1"
  local log_file="$LOG_DIR/seed_${SEED}.log"
  local ckpt_dir="./saved/two_phase_run_tfidf_llm_v3_book_crossing_stratified_seed${SEED}"

  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  log_master ">>> SEED=$SEED"
  log_master ">>> log: $log_file"
  log_master ">>> checkpoint_dir: $ckpt_dir"
  log_master ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  {
    echo "=== seed=$SEED ==="
    echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    python scripts/two_phase_train.py \
      --model SASRecAlignV3 \
      --dataset book-crossing \
      --config_files "sasrec_align_book_crossing_qwen3_stratified_v3.yaml" \
      --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
      --gpu_id "$GPU_ID" \
      --phase_a_grid \
      --align_grid "0.10" \
      --tau_grid "0.05" \
      --backbone_burnin_epochs 0 \
      --burnin_eval_step 2 \
      --phase_a_epochs 20 \
      --phase_a_eval_step 1 \
      --phase_a_valid_metric "MRR@10" \
      --metric_baseline "$METRIC_BASELINE" \
      --metric_gain_threshold 0.01 \
      --lr_text_head 2e-3 \
      --lr_dnn_cross 5e-4 \
      --phase_a_auto_to_b \
      --phase_b_epochs 40 \
      --backbone_lr_scale 0.1 \
      --checkpoint_dir "$ckpt_dir" \
      --seed "$SEED" \
      --variant_features "sasrec,tfidf,llm,qwen3,v3,book_crossing,stratified,seed_${SEED}" \
      --watchdog_disable \
      --save

    echo ""
    echo "结束: $(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee "$log_file"

  log_master ">>> 完成 seed=$SEED  ($(date '+%Y-%m-%d %H:%M:%S'))"
  log_master ""
}

for SEED in $SEEDS; do
  run_one_seed "$SEED"
done

log_master "########################################################################"
log_master "# 全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "# 汇总: $MASTER_LOG"
log_master "########################################################################"
