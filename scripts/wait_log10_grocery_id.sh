#!/usr/bin/env bash
# 等待 log10 Grocery ID-only 完成，并解析 valid MRR@10

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_grocery}"
POLL_SEC="${POLL_SEC:-120}"
LOG_NAME_FILE="$ROOT/logs/.log10_grocery_id_logname_${RUN_ID}.txt"

if [[ ! -f "$LOG_NAME_FILE" ]]; then
  echo "ERROR: missing $LOG_NAME_FILE — run remote_start first"
  exit 1
fi
LOG_NAME="$(cat "$LOG_NAME_FILE")"
REMOTE_LOG="${LOG10_PROJECT}/logs/${LOG_NAME}"

echo ">>> Waiting log10 ID-only: $REMOTE_LOG"

while ssh "$LOG10_SSH" "pgrep -f 'run_log10_grocery_id_only|run_recbole.py.*Grocery' >/dev/null 2>&1"; do
  echo "[$(date '+%H:%M:%S')] log10 still training..."
  sleep "$POLL_SEC"
done

echo ">>> log10 ID-only finished"
ssh "$LOG10_SSH" "tail -30 ${REMOTE_LOG}" || true

MRR="$(ssh "$LOG10_SSH" "python3 - \"${REMOTE_LOG}\" <<'PY'
import re, sys
text = open(sys.argv[1], errors='replace').read()
for pat in [r\"'MRR@10':\\s*([\\d.]+)\", r'\"MRR@10\":\\s*([\\d.]+)', r'MRR@10[\\'\\\"]?:\\s*([\\d.]+)']:
    hits = re.findall(pat, text)
    if hits:
        print(hits[-1])
        break
PY
" 2>/dev/null || true)"

if [[ -n "$MRR" ]]; then
  echo "METRIC_BASELINE=$MRR"
  echo "$MRR" > "$ROOT/logs/.grocery_metric_baseline_${RUN_ID}.txt"
else
  echo "WARN: could not parse MRR@10 from log10"
  exit 1
fi

# 拉回 run_metrics（若有）
rsync -avz "${LOG10_SSH}:${LOG10_PROJECT}/run_metrics/" "$ROOT/run_metrics_log10/" 2>/dev/null || true
