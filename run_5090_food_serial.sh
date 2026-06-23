#!/usr/bin/env bash
# Food 四配置串行流水线：setup → embeddings → ID / TF-IDF / LLM / MV
#
# 与 Beauty/Toys 主表 V3 Aggressive 超参一致；默认 seed=2024。
#
# 一键启动（5090，embedding 已就绪时）:
#   START_FROM=2_id SKIP_SETUP=1 SKIP_EMB=1 \
#   RUN_ID=20260622_food nohup bash run_5090_food_serial.sh \
#     > logs/exp_5090_food_20260622_food.log 2>&1 &
#
# 全量（含 setup + embedding）:
#   RUN_ID=20260622_food nohup bash run_5090_food_serial.sh \
#     > logs/exp_5090_food_20260622_food.log 2>&1 &
#
# 断点续跑:
#   START_FROM=3_tfidf SKIP_SETUP=1 SKIP_EMB=1 RUN_ID=<same> bash run_5090_food_serial.sh
#
# 可选环境变量:
#   GPU_ID=0  SEED=2024  METRIC_BASELINE=0.015  AUTO_METRIC_BASELINE=1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_food}"
LOG_DIR="${LOG_DIR:-logs/5090_food_${RUN_ID}}"
START_FROM="${START_FROM:-0_setup}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_EMB="${SKIP_EMB:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
AUTO_METRIC_BASELINE="${AUTO_METRIC_BASELINE:-1}"

mkdir -p "$LOG_DIR" logs
HISTORY="${HISTORY:-logs/5090_food_${RUN_ID}_history.log}"
MASTER_LOG="$LOG_DIR/00_master.log"

export GPU_ID SEED METRIC_BASELINE

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$RUN_ID] $*" | tee -a "$HISTORY" "$MASTER_LOG"
}

should_run() {
  local target="$1"
  local order=(0_setup 1_emb 2_id 3_tfidf 4_tfidf_llm 5_mv)
  local start_idx=-1 target_idx=-1 i
  for i in "${!order[@]}"; do
    [[ "${order[$i]}" == "$START_FROM" ]] && start_idx=$i
    [[ "${order[$i]}" == "$target" ]] && target_idx=$i
  done
  [[ $start_idx -ge 0 && $target_idx -ge $start_idx ]]
}

extract_baseline_mrr() {
  local logfile="$1"
  python3 - "$logfile" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
patterns = [
    r"'MRR@10':\s*([\d.]+)",
    r'"MRR@10":\s*([\d.]+)',
    r"MRR@10['\"]?:\s*([\d.]+)",
]
for pat in patterns:
    hits = re.findall(pat, text)
    if hits:
        print(hits[-1])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_step() {
  local tag="$1"
  shift
  local logfile="$LOG_DIR/${tag}.log"
  log "▶ START $tag → $logfile"
  local t0=$SECONDS
  {
    echo "=== START: $(date '+%Y-%m-%d %H:%M:%S') ==="
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" "$@"
    echo "=== END:   $(date '+%Y-%m-%d %H:%M:%S') ==="
  } 2>&1 | tee "$logfile"
  log "✅ $tag finished in $((SECONDS - t0))s"
}

log "======== Food Experiment Pipeline ========"
log "RUN_ID=$RUN_ID SEED=$SEED GPU=$GPU_ID METRIC_BASELINE=$METRIC_BASELINE"
log "START_FROM=$START_FROM SKIP_SETUP=$SKIP_SETUP SKIP_EMB=$SKIP_EMB AUTO_METRIC_BASELINE=$AUTO_METRIC_BASELINE"
log "LOG_DIR=$LOG_DIR"

if should_run "0_setup" && [[ "$SKIP_SETUP" != "1" ]]; then
  run_step "0_setup" bash tools/setup_food_dataset.sh
fi

if should_run "1_emb" && [[ "$SKIP_EMB" != "1" ]]; then
  run_step "1_emb" bash tools/gen_text_emb_food_qwen2.5_7b.sh
fi

if should_run "2_id"; then
  run_step "2_id" bash run50epBase_food_stratified.sh
  if [[ "$AUTO_METRIC_BASELINE" == "1" ]]; then
    if mrr="$(extract_baseline_mrr "$LOG_DIR/2_id.log" 2>/dev/null)"; then
      METRIC_BASELINE="$mrr"
      export METRIC_BASELINE
      log "📌 AUTO_METRIC_BASELINE: updated METRIC_BASELINE=$METRIC_BASELINE (from 2_id valid MRR@10)"
    else
      log "⚠️  Could not parse valid MRR@10 from 2_id.log — keep METRIC_BASELINE=$METRIC_BASELINE"
    fi
  fi
fi

if should_run "3_tfidf"; then
  run_step "3_tfidf" bash two_phase_run_tfidf_v3_food_stratified.sh
fi

if should_run "4_tfidf_llm"; then
  run_step "4_tfidf_llm" bash two_phase_run_tfidf_llm_v3_food_stratified.sh
fi

if should_run "5_mv"; then
  run_step "5_mv" bash two_phase_run_multiview_v3_food_stratified_7b.sh
fi

log "======== Food pipeline complete ========"
