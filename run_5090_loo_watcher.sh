#!/usr/bin/env bash
# Watcher: wait for resume_seed2026 to finish, then start Beauty Clean LOO
#
# Usage:
#   nohup bash run_5090_loo_watcher.sh > logs/loo_watcher_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started, waiting for resume_seed2026 to finish..."

while pgrep -f 'run_5090_wsdm_resume_seed2026' > /dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] resume_seed2026 finished. Sleeping 10s before starting LOO..."
sleep 10

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Beauty Clean LOO..."
nohup bash "$ROOT/run_5090_beauty_loo_serial.sh" > logs/beauty_loo_nohup.log 2>&1 &
echo "[$(date '+%Y-%m-%d %H:%M:%S')] LOO started, PID=$!"
