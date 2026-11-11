#!/usr/bin/env bash
# Watcher: wait for Beauty LOO resume to finish, then start Toys LOO
#
# Usage:
#   nohup bash run_5090_toys_loo_watcher.sh > logs/toys_loo_watcher_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Toys LOO watcher started, waiting for beauty_loo_resume to finish..."

while pgrep -f 'run_5090_beauty_loo_resume' > /dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Beauty LOO resume finished. Sleeping 10s before starting Toys LOO..."
sleep 10

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Toys Clean LOO..."
nohup bash "$ROOT/run_5090_toys_loo_serial.sh" > logs/toys_loo_nohup.log 2>&1 &
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Toys LOO started, PID=$!"
