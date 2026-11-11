#!/usr/bin/env bash
# Paper-closing checkpoint re-evaluation only; no training.
#
# Serial order:
#   1) Warm-target main table: Beauty/Toys × ID/TF/LLM/MV × 3 seeds
#   2) UniSRec Beauty/Toys × 3 seeds
#   3) Zero-mask LOVO: Beauty + Toys seed42

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QLOG="logs/paper_retest_min5_queue.log"
mkdir -p logs paper_recsys
: > "$QLOG"
rm -f logs/paper_retest_min5.DONE

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

START=$SECONDS
log "======== PAPER RETEST MIN5 START ========"
log "No training: checkpoint inference only"

log "[1/3] Warm-target 24-checkpoint evaluation"
bash run_5090_post_unisrec_queue.sh

log "[2/3] UniSRec 6-checkpoint evaluation + [3/3] LOVO"
bash run_5090_remaining_p0.sh

python3 - <<'PY' | tee -a "$QLOG"
import csv
from pathlib import Path

checks = [
    (Path("logs/warm_only_eval_min5.csv"), 24),
    (Path("logs/unisrec_gts_test_eval_min5.csv"), 6),
]
for path, expected in checks:
    if not path.exists():
        raise SystemExit(f"VALIDATION FAIL: missing {path}")
    rows = list(csv.DictReader(path.open()))
    errors = [r for r in rows if r.get("error")]
    if len(rows) != expected or errors:
        raise SystemExit(
            f"VALIDATION FAIL: {path} rows={len(rows)}/{expected} errors={len(errors)}"
        )
    print(f"VALIDATION OK: {path} rows={len(rows)}")

for path in [
    Path("paper_recsys/leave_one_view_out_beauty.md"),
    Path("paper_recsys/leave_one_view_out_toys.md"),
]:
    text = path.read_text()
    if "min_train_interactions=5" not in text:
        raise SystemExit(f"VALIDATION FAIL: {path} is not min5")
    print(f"VALIDATION OK: {path} min5")
PY

ELAPSED=$((SECONDS - START))
touch logs/paper_retest_min5.DONE
log "======== PAPER RETEST MIN5 COMPLETE elapsed=${ELAPSED}s ========"
