#!/usr/bin/env bash
# 5090 Grocery 流水线：setup → log10 ID ∥ embedding → TF-IDF / LLM / MV → 汇总
#
# ID-only 在 log10 (1080Ti)；文本特征 + 三文本配置在 5090。
#
# Usage:
#   RUN_ID=20260624_grocery SEED=2024 nohup bash run_5090_grocery_pipeline.sh \
#     > logs/grocery_pipeline_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
source "$ROOT/scripts/log10_env.sh"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_grocery}"
LOG_DIR="${LOG_DIR:-logs/5090_grocery_${RUN_ID}}"
START_FROM="${START_FROM:-0_setup}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_EMB="${SKIP_EMB:-0}"
SKIP_LOG10="${SKIP_LOG10:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
AUTO_METRIC_BASELINE="${AUTO_METRIC_BASELINE:-1}"

mkdir -p "$LOG_DIR" logs run_metrics_log10

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$RUN_ID] $*" | tee -a "$LOG_DIR/00_master.log" logs/grocery_pipeline_history.log
}

gpu_busy() {
  pgrep -f "two_phase_train.py|run_recbole.py" >/dev/null 2>&1
}

wait_gpu() {
  while gpu_busy; do
    log "5090 GPU busy — sleep 120s"
    sleep 120
  done
}

run_step() {
  local tag="$1"
  shift
  local logfile="$LOG_DIR/${tag}.log"
  log "▶ START $tag"
  local t0=$SECONDS
  { echo "=== START $(date) ==="; env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" "$@"; echo "=== END $(date) ==="; } \
    2>&1 | tee "$logfile"
  log "✅ $tag done ($((SECONDS - t0))s)"
}

log "======== Grocery Pipeline (5090 + log10) ========"
log "RUN_ID=$RUN_ID SEED=$SEED SKIP_LOG10=$SKIP_LOG10"

if [[ "$START_FROM" == "0_setup" || "$START_FROM" == "0" ]]; then
  if [[ "$SKIP_SETUP" != "1" && ! -f "logs/.grocery_setup_done" ]]; then
    run_step "0_setup" bash tools/setup_grocery_dataset.sh
    touch logs/.grocery_setup_done
  elif [[ -f "logs/.grocery_setup_done" ]]; then
    log "setup already done (logs/.grocery_setup_done)"
  fi

  if [[ "$SKIP_LOG10" != "1" ]]; then
    if pgrep -f "run_recbole.py.*Grocery" >/dev/null 2>&1 || \
       ssh "$LOG10_SSH" "pgrep -f 'run_recbole.py.*Grocery' >/dev/null 2>&1" 2>/dev/null; then
      log "log10 ID-only already running — skip remote_start"
    elif [[ -f "logs/.log10_grocery_id_logname_${RUN_ID}.txt" ]]; then
      log "log10 ID-only already started for RUN_ID=$RUN_ID — skip remote_start"
    elif bash scripts/remote_start_log10_grocery_id.sh; then
      log "log10 ID-only started in background"
    else
      log "⚠️  log10 unreachable — fallback: ID on 5090"
      SKIP_LOG10=1
    fi
  fi

  if [[ "$SKIP_EMB" != "1" ]]; then
    run_step "1_emb" bash tools/gen_text_emb_grocery_qwen2.5_7b.sh
  fi

  if [[ "$SKIP_LOG10" != "1" ]]; then
    if bash scripts/wait_log10_grocery_id.sh; then
      if [[ -f "logs/.grocery_metric_baseline_${RUN_ID}.txt" ]]; then
        METRIC_BASELINE="$(cat "logs/.grocery_metric_baseline_${RUN_ID}.txt")"
        export METRIC_BASELINE
        log "METRIC_BASELINE from log10: $METRIC_BASELINE"
      fi
    else
      log "⚠️  log10 wait failed — fallback ID on 5090"
      wait_gpu
      run_step "2_id_fallback" bash run50epBase_grocery_stratified.sh
    fi
  else
    wait_gpu
    run_step "2_id" bash run50epBase_grocery_stratified.sh
    if [[ "$AUTO_METRIC_BASELINE" == "1" ]]; then
      if mrr="$(grep -oE "MRR@10['\"]?:[^0-9]*[0-9.]+" "$LOG_DIR/2_id.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$' || true)"; then
        [[ -n "$mrr" ]] && METRIC_BASELINE="$mrr" && export METRIC_BASELINE && log "METRIC_BASELINE=$METRIC_BASELINE"
      fi
    fi
  fi
fi

wait_gpu
run_step "3_tfidf" bash two_phase_run_tfidf_v3_grocery_stratified.sh

wait_gpu
run_step "4_llm" bash two_phase_run_tfidf_llm_v3_grocery_stratified.sh

wait_gpu
run_step "5_mv" bash two_phase_run_multiview_v3_grocery_stratified_7b.sh

log "▶ summarize"
bash tools/summarize_bc_grocery_results.sh >> "$LOG_DIR/99_summarize.log" 2>&1 || true

log "======== Grocery pipeline complete ========"
