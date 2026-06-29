#!/usr/bin/env bash
# 5090 后台 watchdog：Phase 4 text 结束后自动启动 tail，并 kill 可能阻塞的 wait_and_pull。
# 无需半夜人工干预。启动一次即可：
#   nohup bash scripts/5090_auto_tail_watchdog.sh >> logs/5090_auto_tail_watchdog.log 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="${ROOT}/logs/5090_auto_tail_watchdog.log"
DONE="${ROOT}/logs/5090_pipeline_all_complete.done"
TAIL_LOG="${ROOT}/logs/5090_phase4_tail_rebalanced.log"
POLL="${POLL_SEC:-60}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [watchdog] $*" | tee -a "$LOG"; }

phase4_text_done() {
  compgen -G "${ROOT}/saved/gru4rec_multiview_v3_toys_stratified_seed2026/*.pth" >/dev/null \
    && ! pgrep -f "python scripts/two_phase_train.py.*gru4rec.*seed2026" >/dev/null
}

tail_running() {
  pgrep -f "run_5090_phase4_tail_rebalanced.sh" >/dev/null \
    || pgrep -f "run_unisrec_v3_align_balanced_batch_beauty" >/dev/null \
    || pgrep -f "run_5090_gru4rec_baselines_seeds.sh" >/dev/null \
    || pgrep -f "scripts/gru4rec_baseline_run_one.sh" >/dev/null \
    || pgrep -f "python scripts/two_phase_train.py.*gru4rec" >/dev/null
}

log "watchdog started (poll=${POLL}s)"

while [[ ! -f "$DONE" ]]; do
  if grep -q "ALL COMPLETE" "$TAIL_LOG" 2>/dev/null; then
    touch "$DONE"
    log "tail log shows complete — exit"
    break
  fi

  if phase4_text_done; then
    if pgrep -f "wait_and_pull_log10_gru4rec" >/dev/null; then
      log "killing blocking wait_and_pull ..."
      pkill -f "wait_and_pull_log10_gru4rec" || true
      sleep 3
    fi
    if ! tail_running; then
      if phase4_text_done; then
        log "starting run_5090_phase4_tail_rebalanced.sh ..."
        nohup bash "${ROOT}/run_5090_phase4_tail_rebalanced.sh" >> "$TAIL_LOG" 2>&1 &
        sleep 10
      fi
    fi
  fi
  sleep "$POLL"
done

log "watchdog exit (pipeline complete)"
