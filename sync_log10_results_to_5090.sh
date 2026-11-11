#!/usr/bin/env bash
# Sync log10 TS experiment results back to 5090
# Run this on 5090 after log10 ID-only runs complete

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

echo ">>> Syncing log10 TS results → 5090"

RSYNC=(rsync -avz --progress)

# Sync logs
"${RSYNC[@]}" "${LOG10_SSH}:${LOG10_PROJECT}/logs/ts_*" "$ROOT/logs/" 2>/dev/null || echo "No log10 TS logs yet"

# Sync saved checkpoints
for DIR in ts_amazon_beauty_id_only ts_amazon_toys_and_games_id_only ts_amazon_grocery_and_gourmet_food_id_only; do
  for SEED in 2024 2025 42; do
    SRC="${LOG10_SSH}:${LOG10_PROJECT}/saved/${DIR}_seed${SEED}/"
    DST="$ROOT/saved/${DIR}_seed${SEED}/"
    if ssh "$LOG10_SSH" "test -d ${LOG10_PROJECT}/saved/${DIR}_seed${SEED}" 2>/dev/null; then
      mkdir -p "$DST"
      "${RSYNC[@]}" "$SRC" "$DST"
    fi
  done
done

# Sync run_metrics
"${RSYNC[@]}" "${LOG10_SSH}:${LOG10_PROJECT}/run_metrics/" "$ROOT/run_metrics/" 2>/dev/null || echo "No run_metrics"

echo "✅ log10 → 5090 result sync done"
