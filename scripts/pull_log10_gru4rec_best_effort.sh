#!/usr/bin/env bash
# Best-effort rsync log10 GRU4Rec artifacts → 5090 (no wait).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"
REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log ">>> rsync log10 → 5090 (best effort)"
rsync -avz "${REMOTE}/saved/gru4rec_"* "${ROOT}/saved/" 2>/dev/null || true
rsync -avz "${REMOTE}/run_metrics/" "${ROOT}/run_metrics/" 2>/dev/null || true
rsync -avz "${REMOTE}/logs/log10_gru4rec_"*.log "${ROOT}/logs/" 2>/dev/null || true
log "✅ pull done (no wait)"
