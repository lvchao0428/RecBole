#!/usr/bin/env bash
# Wait for UniSRec resume to finish, then start post queue (warm-only + tables).
#
# Usage:
#   nohup bash run_5090_post_unisrec_watcher.sh > logs/post_unisrec_watcher_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher: waiting for run_5090_unisrec_gts_resume ..."

while pgrep -f 'run_5090_unisrec_gts_resume' >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] UniSRec resume finished. Starting post queue in 10s..."
sleep 10

nohup bash "$ROOT/run_5090_post_unisrec_queue.sh" > logs/post_unisrec_nohup.log 2>&1 &
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Post queue started, PID=$!"
