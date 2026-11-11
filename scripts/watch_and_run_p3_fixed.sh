#!/usr/bin/env bash
# Wait for the in-flight 0817 Hybrid / Grocery CKA job, then start the
# ordered paper rerun queue. Does not kill running training.

set -euo pipefail
ROOT="/home/charlie/project/RecBole"
LOG="$ROOT/logs/p3_watcher_0818.log"
echo "[$(date)] watcher started; waiting for 0817 queue to finish." >> "$LOG"

while pgrep -f "run_final_supplement_0817.sh" >/dev/null 2>&1; do
  sleep 120
done
while pgrep -f "two_phase_train.py" >/dev/null 2>&1; do
  sleep 60
done
while pgrep -f "build_item_text_emb_qwen2.5_7b.py" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date)] Queue idle. Launching paper rerun P0–P4." >> "$LOG"
cd "$ROOT"
bash "$ROOT/scripts/run_paper_rerun_queue_0818.sh" >> "$ROOT/logs/paper_rerun_queue_0818_nohup.log" 2>&1
echo "[$(date)] paper rerun queue finished exit=$?" >> "$LOG"
