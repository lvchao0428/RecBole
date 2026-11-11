#!/usr/bin/env bash
# Remaining P0 for 0723 guidance — 5090 serial queue
#
#   1) UniSRec GTS true test(+valid) re-eval on 6 existing ckpts  (B2 fix)
#   2) Leave-one-view-out on Beauty + Toys MV GTS ckpts           (B1 fix)
#
# Usage:
#   nohup bash run_5090_remaining_p0.sh > logs/remaining_p0_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
QLOG="logs/remaining_p0_queue.log"
mkdir -p logs paper_recsys
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

log "======== Remaining P0 START (GPU=$GPU_ID) ========"

# -------- 1) UniSRec true test eval --------
LIST="logs/unisrec_eval_list.txt"
cat > "$LIST" <<'EOF'
saved/unisrec_beauty_gts_seed42
saved/unisrec_beauty_gts_seed2024
saved/unisrec_beauty_gts_seed2026
saved/unisrec_toys_gts_seed42
saved/unisrec_toys_gts_seed2024
saved/unisrec_toys_gts_seed2026
EOF

log "======== [1/2] UniSRec test(+valid) re-eval ========"
python scripts/eval_ckpt_test.py \
  --ckpt_list "$LIST" \
  --out logs/unisrec_gts_test_eval.csv \
  --gpu_id "$GPU_ID" \
  --also_valid \
  > logs/unisrec_gts_test_eval.log 2>&1 || log "WARN: UniSRec eval had errors (see logs/unisrec_gts_test_eval.log)"

if [[ -f logs/unisrec_gts_test_eval.csv ]]; then
  log "UniSRec CSV:"
  tee -a "$QLOG" < logs/unisrec_gts_test_eval.csv
  # Rewrite corrected summary CSV (compatible columns)
  python3 - <<'PY' || true
import csv
from pathlib import Path
src = Path("logs/unisrec_gts_test_eval.csv")
dst = Path("logs/unisrec_gts_results_corrected.csv")
rows = list(csv.DictReader(src.open()))
with dst.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["tag","model","dataset","seed","valid_mrr","test_mrr","test_ndcg","test_hit"])
    w.writeheader()
    for r in rows:
        if r.get("error"):
            continue
        w.writerow({
            "tag": r.get("tag",""),
            "model": r.get("model","UniSRec"),
            "dataset": r.get("dataset",""),
            "seed": r.get("seed",""),
            "valid_mrr": r.get("valid_mrr@10", ""),
            "test_mrr": r.get("test_mrr@10", ""),
            "test_ndcg": r.get("test_ndcg@10", ""),
            "test_hit": r.get("test_hit@10", ""),
        })
print(f"Wrote {dst}")
PY
  cp -f logs/unisrec_gts_test_eval.csv paper_recsys/ 2>/dev/null || true
  cp -f logs/unisrec_gts_results_corrected.csv paper_recsys/ 2>/dev/null || true
fi

# -------- 2) Leave-one-view-out (fixed zero-mask) --------
log "======== [2/2] Leave-one-view-out (Beauty + Toys) ========"

# Prefer GTS Main stop-gate / wrap MV ckpts (fair protocol)
BEAUTY_CKPT="saved/sg_beauty_mv_lr5e4_do0.3_seed42"
TOYS_CKPT="saved/wrap_toys_mv_lr5e4_do0.3_seed42"

if [[ ! -d "$BEAUTY_CKPT" ]]; then
  BEAUTY_CKPT="saved/loo_beauty_mv_lr5e4_do0.3_seed42"
fi
if [[ ! -d "$TOYS_CKPT" ]]; then
  TOYS_CKPT="saved/loo_toys_mv_lr5e4_do0.3_seed42"
fi

log "Beauty LOVO ckpt=$BEAUTY_CKPT"
python scripts/analyze_leave_one_view_out.py \
  --ckpt "$BEAUTY_CKPT" \
  --dataset Amazon_Beauty \
  --output paper_recsys/leave_one_view_out_beauty.md \
  --gpu_id "$GPU_ID" \
  > logs/lovo_beauty.log 2>&1 || log "WARN: Beauty LOVO failed (see logs/lovo_beauty.log)"

log "Toys LOVO ckpt=$TOYS_CKPT"
python scripts/analyze_leave_one_view_out.py \
  --ckpt "$TOYS_CKPT" \
  --dataset Amazon_Toys_and_Games \
  --output paper_recsys/leave_one_view_out_toys.md \
  --gpu_id "$GPU_ID" \
  > logs/lovo_toys.log 2>&1 || log "WARN: Toys LOVO failed (see logs/lovo_toys.log)"

# brief tail into queue log
for f in logs/lovo_beauty.log logs/lovo_toys.log; do
  if [[ -f "$f" ]]; then
    log "--- tail $f ---"
    tail -30 "$f" | tee -a "$QLOG" || true
  fi
done

touch logs/remaining_p0.DONE
log "======== Remaining P0 COMPLETE ========"
log "Artifacts:"
log "  logs/unisrec_gts_test_eval.csv"
log "  logs/unisrec_gts_results_corrected.csv"
log "  paper_recsys/leave_one_view_out_beauty.md"
log "  paper_recsys/leave_one_view_out_toys.md"
