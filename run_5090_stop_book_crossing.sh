#!/usr/bin/env bash
# 停止 book-crossing 相关队列，记录现状后切换至 Food 实验
#
# Usage:
#   bash run_5090_stop_book_crossing.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SNAP="paper_recsys/book_crossing_status_snapshot_$(date +%Y%m%d).md"
mkdir -p logs

{
  echo "# Book-Crossing 停止前快照"
  echo ""
  echo "> 时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo ""
  echo "## 运行中进程"
  pgrep -af "run_5090_bc_aligned|run_5090_queue|book_crossing|two_phase.*book_crossing" || echo "(none)"
  echo ""
  echo "## 最近 bc_aligned history (tail 30)"
  tail -30 logs/5090_bc_aligned_*_history.log 2>/dev/null || echo "(no history)"
  echo ""
  echo "## saved/ bc 相关目录"
  ls -d saved/*book_crossing* saved/*bc_aligned* 2>/dev/null || echo "(none)"
} | tee "$SNAP" logs/stop_bc_snapshot.log

PATTERNS=(
  "run_5090_bc_aligned_ablation_serial.sh"
  "run_5090_queue_sasrecalign_baseline.sh"
  "run_5090_queue_mv_g1_g4.sh"
  "run_5090_queue_bc_mv_phaseb50.sh"
  "two_phase_run_multiview_v3_book_crossing"
  "two_phase_run_tfidf.*book_crossing"
  "two_phase_train.py.*book-crossing"
)

for pat in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "$pat" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Stopping: $pat → $pids"
    kill $pids 2>/dev/null || true
  fi
done

sleep 2
echo ""
echo "Remaining bc processes:"
pgrep -af "book_crossing|bc_aligned" || echo "(none)"
echo ""
echo "Snapshot saved: $SNAP"
