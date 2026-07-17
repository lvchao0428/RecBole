#!/usr/bin/env bash
# Sync 5090 TS experiment results to local machine
# Run from local machine

set -euo pipefail

REMOTE="charlie@www.ultrapp.online"
REMOTE_DIR="/home/charlie/project/RecBole"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> Syncing 5090 TS results → local"

RSYNC=(rsync -avz --progress)

# Sync logs
mkdir -p "$LOCAL_DIR/logs"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/logs/ts_*" "$LOCAL_DIR/logs/" 2>/dev/null || echo "No TS logs yet"

# Sync run_metrics
mkdir -p "$LOCAL_DIR/run_metrics"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/run_metrics/" "$LOCAL_DIR/run_metrics/" 2>/dev/null || echo "No run_metrics"

echo "✅ 5090 → local result sync done"
