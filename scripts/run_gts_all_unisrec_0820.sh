#!/usr/bin/env bash
# GTS-All warm re-eval on UniSRec ckpts (Beauty gts_seed*, Toys gts_ts_seed*); no retraining.
# Purpose: replace the all-target / catalog-fit UniSRec GTS-All numbers in appendix app:unisrec.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LIST="$ROOT/logs/gts_all_unisrec_ckpt_list.txt"
OUT="$ROOT/paper_recsys/gts_all_eval_unisrec_ts.csv"
LOG="$ROOT/logs/gts_all_unisrec_0820.log"
: > "$LOG"

{
  for SEED in 42 2024 2026; do echo "saved/unisrec_beauty_gts_seed${SEED}"; done
  for SEED in 42 2024 2026; do echo "saved/unisrec_toys_gts_ts_seed${SEED}"; done
} > "$LIST"

echo "[$(date "+%F %H:%M:%S")] GTS-All warm UniSRec START" | tee -a "$LOG"
python3 "$ROOT/scripts/eval_warm_targets_only.py" \
  --ckpt_list "$LIST" --out "$OUT" --gpu_id 0 \
  --min_train_interactions 5 --skip_valid \
  >> "$LOG" 2>&1
echo "[$(date "+%F %H:%M:%S")] DONE -> $OUT" | tee -a "$LOG"
touch "$ROOT/logs/gts_all_unisrec_0820.DONE"
