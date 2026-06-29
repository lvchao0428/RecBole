#!/usr/bin/env bash
# Start log10 GRU4Rec resume (from seed=2024) on 1080Ti. Run on 5090.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_NAME="log10_resume_after_poweroff_${RUN_ID}.log"
PID_FILE="$ROOT/logs/.log10_gru4rec_resume_pid_${RUN_ID}.txt"

_running() {
  ssh "$LOG10_SSH" "pgrep -f 'python scripts/two_phase_train.py.*GRU4Rec' >/dev/null \
    || pgrep -f 'bash run_log10_gru4rec_baselines' >/dev/null \
    || pgrep -f 'bash run_log10_resume_after_poweroff' >/dev/null"
}

if _running; then
  echo ">>> log10 GRU4Rec already running — skip"
  ssh "$LOG10_SSH" "pgrep -af 'python scripts/two_phase_train.py|bash run_log10_' | grep -v pgrep | head -3"
  exit 0
fi

echo ">>> Starting log10 GRU4Rec resume on $LOG10_SSH"
ssh -n "$LOG10_SSH" "mkdir -p ${LOG10_PROJECT}/logs"
ssh -n "$LOG10_SSH" "cd ${LOG10_PROJECT} && source scripts/recbole_env.sh && nohup bash run_log10_resume_after_poweroff.sh \
  > logs/${LOG_NAME} 2>&1 </dev/null & echo \$!" | tee "$PID_FILE"

echo "log10 PID: $(cat "$PID_FILE")"
echo "log10 log: ${LOG10_PROJECT}/logs/${LOG_NAME}"
echo "$LOG_NAME" > "$ROOT/logs/.log10_gru4rec_resume_logname.txt"
