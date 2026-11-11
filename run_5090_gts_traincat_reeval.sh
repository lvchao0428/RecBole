#!/usr/bin/env bash
#
# Re-evaluate Beauty/Toys GTS-Last + GTS-Random with train-catalog candidate mask.
# Overwrites gts_last_eval.csv / gts_random_eval.csv (backs up previous Random).
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/recbole_env.sh" 2>/dev/null || true

GPU_ID="${GPU_ID:-0}"
QLOG="logs/gts_traincat_reeval_queue.log"
CKPT_LIST="logs/gts_last_ckpt_list.txt"
mkdir -p logs paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

if [[ ! -f "$CKPT_LIST" ]]; then
  log "FATAL: missing $CKPT_LIST"
  exit 1
fi

log "======== GTS train-catalog re-eval START (GPU=$GPU_ID) ========"

log "---- GTS-Last (train_catalog candidates) ----"
python3 scripts/eval_gts_last.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_last_eval.csv \
  --min_train_interactions 5 \
  --gpu_id "$GPU_ID" 2>&1 | tee logs/gts_last_eval_traincat.log
cp -f logs/gts_last_eval.csv paper_recsys/gts_last_eval.csv

log "---- GTS-Random (train_catalog candidates, force) ----"
python3 scripts/eval_gts_random.py \
  --ckpt_list "$CKPT_LIST" \
  --out logs/gts_random_eval.csv \
  --min_train_interactions 5 \
  --sample_seed 42 \
  --gpu_id "$GPU_ID" \
  --force 2>&1 | tee logs/gts_random_eval_traincat.log
cp -f logs/gts_random_eval.csv paper_recsys/gts_random_eval.csv

if [[ -f scripts/build_main_tables_gts.py ]]; then
  log "Building main_table_gts_last_random.md ..."
  python3 scripts/build_main_tables_gts.py \
    --last paper_recsys/gts_last_eval.csv \
    --random paper_recsys/gts_random_eval.csv \
    --out_md paper_recsys/main_table_gts_last_random.md \
    --out_csv paper_recsys/main_table_gts_last_random.csv \
    >> "$QLOG" 2>&1 || log "WARN: build_main_tables_gts.py failed (non-fatal)"
fi

# Spot-check: warm_candidate_scope must be train_catalog
python3 - <<'PY' | tee -a "$QLOG"
import csv
from pathlib import Path
for name in ("logs/gts_last_eval.csv", "logs/gts_random_eval.csv"):
    p = Path(name)
    rows = list(csv.DictReader(p.open()))
    scopes = {r.get("warm_candidate_scope") for r in rows}
    print(f"{name}: n={len(rows)} scopes={scopes}")
    if scopes != {"train_catalog"}:
        raise SystemExit(f"BAD scope in {name}: {scopes}")
print("OK: all rows warm_candidate_scope=train_catalog")
PY

touch logs/gts_traincat_reeval.DONE
log "======== GTS train-catalog re-eval COMPLETE ========"
