#!/usr/bin/env bash
# Master GTS experiment orchestrator — runs all phases sequentially on 5090
# P2 (Beauty 3-seed text) → P3 (Toys+Grocery text) → P4 (UniSRec) → P5 (Ablation)
#
# ID-only runs on log10 should be started separately via:
#   ssh charlie@192.168.0.107 "cd ~/project/RecBole && nohup bash run_log10_ts_id_only.sh > logs/ts_id_only_nohup.log 2>&1 &"
#
# Usage (on 5090):
#   nohup bash run_5090_ts_all_phases.sh \
#     > logs/ts_all_phases_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="logs/ts_all_phases.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== WSDM GTS Full Pipeline ========"

# Phase 2: Beauty 3-seed text models
log "===== PHASE 2: Beauty 3-seed text ====="
bash run_5090_ts_beauty_3seed.sh
log "===== PHASE 2 COMPLETE ====="

# Phase 3: Toys + Grocery
log "===== PHASE 3: Toys + Grocery ====="
bash run_5090_ts_toys_grocery.sh
log "===== PHASE 3 COMPLETE ====="

# Phase 4: UniSRec
log "===== PHASE 4: UniSRec ====="
bash run_5090_ts_unisrec.sh
log "===== PHASE 4 COMPLETE ====="

# Phase 5: Ablation
log "===== PHASE 5: Ablation ====="
bash run_5090_ts_ablation.sh
log "===== PHASE 5 COMPLETE ====="

log "======== ALL PHASES COMPLETE ========"
log "Don't forget to sync log10 results: bash scripts/sync_log10_results.sh"
