#!/usr/bin/env bash
# Wait for the current two_phase_train job to finish, then stop the pipeline
# before the next queued run. Writes a checkpoint doc for post-reboot resume.
#
# Usage (5090):
#   nohup bash scripts/pipeline_pause_after_current.sh \
#     >> logs/pipeline_pause_nohup.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="$ROOT/paper_recsys/pipeline_checkpoint_20260627.md"
LOG="$ROOT/logs/pipeline_pause.log"
POLL_SEC="${POLL_SEC:-30}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Pipeline pause watcher started ========"
log "Will stop run_5090_phase4_text_resume.sh after current two_phase_train exits."

while pgrep -f "two_phase_train.py" >/dev/null 2>&1; do
  sleep "$POLL_SEC"
done

log "Current training job finished."

PIPE_PID=""
while read -r pid cmd; do
  [[ "$cmd" == *"pipeline_pause_after_current"* ]] && continue
  if [[ "$cmd" == *"run_5090_phase4_text_resume"* ]]; then
    PIPE_PID="$pid"
    break
  fi
done < <(pgrep -af "run_5090_phase4_text_resume" || true)

if [[ -n "$PIPE_PID" ]]; then
  log "Stopping pipeline PID=$PIPE_PID before next job..."
  kill "$PIPE_PID" 2>/dev/null || true
  sleep 3
  kill -9 "$PIPE_PID" 2>/dev/null || true
fi

mkdir -p "$ROOT/paper_recsys"
cat > "$CHECKPOINT" <<EOF
# Pipeline Checkpoint（断电暂停 · $(date '+%Y-%m-%d %H:%M:%S')）

> **原因**: 计划断电（修灯泡）· 在当前训练 job 完成后自动暂停队列  
> **5090 下一任务**: Toys GRU4Rec LLM+MV seed=2025  
> **log10 下一任务**: 若同步断电，从 Beauty seed=2024 重跑（当前 job 若未完成需检查 checkpoint）

## 5090 已完成（Phase 4 text · 截至暂停点）

| Seed | Beauty LLM | Beauty MV | Toys LLM | Toys MV |
|------|------------|-----------|----------|---------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024 | ✅ | ✅ | ✅ | ✅ |
| 2025 | ✅ | ✅（暂停前刚完成） | 📋 | 📋 |
| 2026 | 📋 | 📋 | 📋 | 📋 |

**进度**: 5090 **9/16** text runs 完成（Beauty MV 2025 为暂停边界）

## 5090 恢复命令（重启后）

\`\`\`bash
cd ~/project/RecBole
source scripts/recbole_env.sh
nohup bash run_5090_phase4_text_resume_from_toys2025.sh \
  >> logs/phase4_resume_from_toys2025_nohup.log 2>&1 &
tail -f logs/phase4_resume_from_toys2025_nohup.log
\`\`\`

队列后续: Toys 2025 → Beauty/Toys 2026 → wait log10 → Phase 5 UniSRec balanced

## log10 恢复命令（若 log10 也断电）

\`\`\`bash
cd ~/project/RecBole
source scripts/recbole_env.sh
nohup bash run_log10_gru4rec_baselines_from_2024.sh \
  >> logs/log10_gru4rec_resume_from_2024_nohup.log 2>&1 &
\`\`\`

> log10 当前 job（Beauty TF-IDF seed=2024）若被中途打断，该 run 需重跑；脚本会从 seed=2024 完整重跑 beauty+toys 块。

## 监控

\`\`\`bash
# 5090
tail -f logs/pipeline_pause.log
cat paper_recsys/pipeline_checkpoint_20260627.md
\`\`\`
EOF

log "Checkpoint written: $CHECKPOINT"
log "======== Safe to power off 5090 now ========"
