#!/usr/bin/env bash
# 5090 Phase 4 收尾（自包含 v3）：
#   5090 跑完后 Phase 4 + Phase 5 + 全部剩余 baseline 均完成，不阻塞等 log10。
#   log10 多跑重复任务无妨（checkpoint skip 自动去重）。
#
# 顺序:
#   1. pull log10（best effort）
#   2. Phase 5 UniSRec balanced
#   3. baseline seed=2024/2025/2026（skip 已有 ckpt）
#   4. pull log10 + collect
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
LOG="${PHASE4_TAIL_LOG:-logs/5090_phase4_tail_rebalanced.log}"
DONE_MARKER="${ROOT}/logs/5090_pipeline_all_complete.done"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [[ -f "$DONE_MARKER" ]]; then
  log "======== already complete ($DONE_MARKER) — skip ========"
  exit 0
fi

log "======== 5090 Phase 4 tail (self-contained v3) ========"

log ">>> pull log10 (best effort, no wait)"
bash "$ROOT/scripts/pull_log10_gru4rec_best_effort.sh" >> "$LOG" 2>&1 || true

phase5_done() {
  compgen -G "${ROOT}/saved/unisrec_align_v3_beauty_stratified_balanced_seed2024/*.pth" >/dev/null \
    && compgen -G "${ROOT}/saved/unisrec_align_multiview_v3_beauty_stratified_balanced_seed2024/*.pth" >/dev/null
}

if phase5_done; then
  log ">>> Phase 5 skip (UniSRec balanced ckpts exist)"
else
  log ">>> Phase 5: UniSRec Align/MV balanced (Beauty seed=2024)"
  t0=$SECONDS
  env GPU_ID="$GPU_ID" SEED=2024 \
    bash "$ROOT/run_unisrec_v3_align_balanced_batch_beauty.sh" \
    >> "logs/unisrec_align_balanced_batch_beauty_nohup.log" 2>&1
  log "  ✅ Phase 5 UniSRec balanced done ($((SECONDS - t0))s)"
fi

log ">>> 5090 GRU4Rec baselines seed=2024 2025 2026 (5090 aggressive; skip existing ckpt)"
t0=$SECONDS
env GPU_ID="$GPU_ID" TRAIN_BATCH_SIZE=512 EVAL_BATCH_SIZE=512 \
  bash "$ROOT/run_5090_gru4rec_baselines_seeds.sh" 2024 2025 2026 >> "$LOG" 2>&1
log "  ✅ 5090 baselines done ($((SECONDS - t0))s)"

log ">>> final pull log10 + collect"
bash "$ROOT/scripts/pull_log10_gru4rec_best_effort.sh" >> "$LOG" 2>&1 || true
bash "$ROOT/scripts/collect_gru4rec_phase4_results.sh" >> "$LOG" 2>&1

touch "$DONE_MARKER"
log "======== ALL COMPLETE — 5090 pipeline done ($DONE_MARKER) ========"
