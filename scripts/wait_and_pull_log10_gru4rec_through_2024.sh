#!/usr/bin/env bash
# Wait until log10 finishes GRU4Rec baselines through seed=2024, then rsync to 5090.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"
REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Waiting for log10 marker: logs/log10_gru4rec_through_2024.done ..."
while ! ssh "$LOG10_SSH" "test -f ${LOG10_PROJECT}/logs/log10_gru4rec_through_2024.done"; do
  if ssh "$LOG10_SSH" "grep -q 'through 2024 complete' ${LOG10_PROJECT}/logs/log10_gru4rec_baselines.log 2>/dev/null"; then
    break
  fi
  # Toys TF-IDF 2024 is the last log10 job
  ttf=$(ssh "$LOG10_SSH" "ls ${LOG10_PROJECT}/saved/gru4rec_tfidf_v3_toys_stratified_seed2024/*.pth 2>/dev/null | wc -l")
  if [[ "$ttf" -gt 0 ]] && ! ssh "$LOG10_SSH" "pgrep -f 'python scripts/two_phase_train.py.*GRU4Rec' >/dev/null"; then
    log "log10 seed=2024 toys TF-IDF checkpoint found and idle — proceed"
    break
  fi
  sleep 120
done

log ">>> rsync log10 → 5090 (gru4rec saved + run_metrics + logs)"
rsync -avz "${REMOTE}/saved/gru4rec_"* "${ROOT}/saved/" 2>/dev/null || true
rsync -avz "${REMOTE}/run_metrics/" "${ROOT}/run_metrics/" 2>/dev/null || true
rsync -avz "${REMOTE}/logs/log10_gru4rec_"*.log "${ROOT}/logs/" 2>/dev/null || true
rsync -avz "${REMOTE}/logs/5090_gru4rec_"*.log "${ROOT}/logs/" 2>/dev/null || true

log "✅ log10 GRU4Rec through-2024 results pulled to 5090"
