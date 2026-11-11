#!/usr/bin/env bash
# Retry UniSRec true test(+valid) after np.float patch.
# Waits for current remaining_p0 / LOVO to finish, then re-evals 6 UniSRec ckpts.
#
# Usage:
#   nohup bash run_5090_unisrec_test_retry.sh > logs/unisrec_test_retry_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
QLOG="logs/unisrec_test_retry_queue.log"
mkdir -p logs paper_recsys
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

log "Watcher: waiting for remaining_p0 / LOVO to finish..."
while pgrep -f 'run_5090_remaining_p0|analyze_leave_one_view_out' >/dev/null 2>&1; do
  sleep 60
done
log "Prior jobs finished. Sleep 10s, then UniSRec retry."
sleep 10

LIST="logs/unisrec_eval_list.txt"
cat > "$LIST" <<'EOF'
# UniSRec GTS — trained 7/28–7/30, same protocol as main table
# model=UniSRec, TS[0.8,0.1,0.1], min5, only_phase_a 50ep
saved/unisrec_beauty_gts_seed42
saved/unisrec_beauty_gts_seed2024
saved/unisrec_beauty_gts_seed2026
saved/unisrec_toys_gts_seed42
saved/unisrec_toys_gts_seed2024
saved/unisrec_toys_gts_seed2026
EOF

log "======== UniSRec test(+valid) retry START ========"
log "Ckpts: Beauty/Toys × seeds {42,2024,2026} (UniSRec GTS)"

python scripts/eval_ckpt_test.py \
  --ckpt_list "$LIST" \
  --out logs/unisrec_gts_test_eval_min5.csv \
  --gpu_id "$GPU_ID" \
  --min_train_interactions 5 \
  --also_valid \
  > logs/unisrec_gts_test_eval_min5.log 2>&1

python3 - <<'PY' || true
import csv
from pathlib import Path
src = Path("logs/unisrec_gts_test_eval_min5.csv")
dst = Path("logs/unisrec_gts_results_min5.csv")
rows = list(csv.DictReader(src.open()))
errs = [r for r in rows if r.get("error")]
ok = [r for r in rows if not r.get("error")]
print(f"ok={len(ok)} err={len(errs)}")
with dst.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["tag","model","dataset","seed","valid_mrr","test_mrr","test_ndcg","test_hit"])
    w.writeheader()
    for r in ok:
        w.writerow({
            "tag": r.get("tag",""),
            "model": r.get("model","UniSRec"),
            "dataset": r.get("dataset",""),
            "seed": r.get("seed",""),
            "valid_mrr": r.get("valid_mrr@10",""),
            "test_mrr": r.get("test_mrr@10",""),
            "test_ndcg": r.get("test_ndcg@10",""),
            "test_hit": r.get("test_hit@10",""),
        })
print(f"Wrote {dst}")
PY

cp -f logs/unisrec_gts_test_eval_min5.csv paper_recsys/ 2>/dev/null || true
cp -f logs/unisrec_gts_results_min5.csv paper_recsys/ 2>/dev/null || true
tee -a "$QLOG" < logs/unisrec_gts_results_min5.csv || true

touch logs/unisrec_test_retry.DONE
log "======== UniSRec test retry COMPLETE ========"
