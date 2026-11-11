#!/usr/bin/env bash
# Grocery GTS-Random for the two missing target-sampling seeds (123, 456).
# Main table claims a 3-seed Random average; Grocery currently has only seed 42.
# Eval only, no retraining.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LIST="$ROOT/logs/grocery_random_ckpt_list.txt"
LOG="$ROOT/logs/grocery_random_seeds_0819.log"
: > "$LOG"

{
  for SEED in 42 2024 2026; do
    for M in id_lr1e4_do0.1 tf_lr5e4_do0.3 llm_lr1e4_do0.1 mv_lr5e4_do0.3; do
      echo "saved/wrap_grocery_${M}_seed${SEED}"
    done
  done
  for SEED in 42 2024 2026; do
    echo "saved/unisrec_grocery_gts_seed${SEED}"
  done
} > "$LIST"

for SAMPLE_SEED in 123 456; do
  OUT="$ROOT/paper_recsys/gts_random_eval_grocery_seed${SAMPLE_SEED}.csv"
  echo "[$(date '+%F %H:%M:%S')] Grocery GTS-Random sample_seed=${SAMPLE_SEED}" | tee -a "$LOG"
  python3 "$ROOT/scripts/eval_gts_random.py" \
    --ckpt_list "$LIST" --out "$OUT" \
    --min_train_interactions 5 --sample_seed "$SAMPLE_SEED" \
    --gpu_id 0 >> "$LOG" 2>&1
  echo "[$(date '+%F %H:%M:%S')]   -> $OUT" | tee -a "$LOG"
done

touch "$ROOT/logs/grocery_random_seeds_0819.DONE"
echo "[$(date '+%F %H:%M:%S')] Grocery Random seeds DONE" | tee -a "$LOG"
