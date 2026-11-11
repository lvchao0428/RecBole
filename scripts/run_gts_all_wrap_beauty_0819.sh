#!/usr/bin/env bash
# GTS-All warm re-eval on cutoff-fit wrap_beauty TF/LLM/MV (no retraining).
# Replaces catalog-fit sg_beauty_* numbers in appendix tab:gts-all.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LIST="$ROOT/logs/gts_all_wrap_beauty_ckpt_list.txt"
OUT="$ROOT/paper_recsys/gts_all_eval_beauty_ts.csv"
LOG="$ROOT/logs/gts_all_wrap_beauty_0819.log"
: > "$LOG"

{
  for M in tf_lr5e4_do0.3 llm_lr1e4_do0.1 mv_lr5e4_do0.3; do
    for SEED in 42 2024 2026; do
      echo "saved/wrap_beauty_${M}_seed${SEED}"
    done
  done
} > "$LIST"

echo "[$(date '+%F %H:%M:%S')] GTS-All warm wrap_beauty START" | tee -a "$LOG"
python3 "$ROOT/scripts/eval_warm_targets_only.py" \
  --ckpt_list "$LIST" --out "$OUT" --gpu_id 0 \
  --min_train_interactions 5 --skip_valid \
  >> "$LOG" 2>&1
echo "[$(date '+%F %H:%M:%S')] GTS-All warm wrap_beauty DONE -> $OUT" | tee -a "$LOG"
touch "$ROOT/logs/gts_all_wrap_beauty_0819.DONE"
