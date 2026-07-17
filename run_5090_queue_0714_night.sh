#!/usr/bin/env bash
# 5090 夜间队列 — 2026-07-13/14
# 接在 run_beauty_full_boost_v2.sh 之后自动执行：
#   Step 1: 等待 boost+Cross 完成
#   Step 2: boost+no-Cross 4-config (~5.5h)  ← 今晚/明早最重要
#   Step 3: UniSRec Beauty seed=2025 × 3 configs (~2h)
#
# Usage (5090):
#   nohup bash run_5090_queue_0714_night.sh > logs/queue_0714_night_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="logs/queue_0714_night.log"
mkdir -p logs
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Queue 0714 Night Started ========"
log "Waiting for run_beauty_full_boost_v2.sh to finish ..."

while pgrep -f "bash run_beauty_full_boost_v2.sh" >/dev/null 2>&1; do
  sleep 60
done
# also wait if child python still running from that script
while pgrep -f "ts_beauty_fb_.*seed2025" >/dev/null 2>&1; do
  sleep 30
done

log "=== Step 1: boost+Cross complete ==="

log "=== Step 2: boost+no-Cross (cb3+infer, seed=2025) ==="
bash run_beauty_full_boost_nocross_v2.sh >> logs/beauty_full_boost_nocross_v2_nohup.log 2>&1
log "=== Step 2 DONE ==="

log "=== Step 3: UniSRec Beauty seed=2025 (min5) ==="
bash run_5090_ts_unisrec_seed2025.sh >> logs/ts_unisrec_seed2025_nohup.log 2>&1
log "=== Step 3 DONE ==="

log "=== Collecting ablation tables ==="
python3 scripts/collect_p2_ablation.py 2>/dev/null | tee -a "$LOG" || true

log "======== Queue 0714 Night COMPLETE ========"
