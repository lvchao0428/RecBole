#!/usr/bin/env bash
# Watch for GPU to become free (< 1GB used), then launch final supplement.
LOG="/home/charlie/project/RecBole/logs/gpu_watcher_0817.log"
echo "[$(date)] Watcher started. Waiting for GPU to be free..." >> "$LOG"

while true; do
  MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$MEM_USED" ]; then
    echo "[$(date)] nvidia-smi failed, retrying..." >> "$LOG"
    sleep 60
    continue
  fi
  if [ "$MEM_USED" -lt 1000 ]; then
    echo "[$(date)] GPU free (${MEM_USED}MiB used). Launching experiments." >> "$LOG"
    cd /home/charlie/project/RecBole
    nohup bash scripts/run_final_supplement_0817.sh >> logs/final_supplement_0817_nohup.log 2>&1 &
    echo "[$(date)] Launched PID=$!" >> "$LOG"
    exit 0
  fi
  sleep 300
done
