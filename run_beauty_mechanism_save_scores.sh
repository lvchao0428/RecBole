#!/usr/bin/env bash
# Beauty mechanism: eval-only save_test_scores for ID / TF-IDF / MV (seed=2024).
# Uses existing checkpoints — no re-training (~30–45 min each eval).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
SCORES_DIR="${SCORES_DIR:-ablation_study_doc/scores}"
PERUSER_DIR="${PERUSER_DIR:-saved/peruser/mechanism}"
LOG="${MECHANISM_SCORES_LOG:-logs/mechanism_save_scores_beauty.log}"
mkdir -p logs "$SCORES_DIR" "$PERUSER_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

latest_ckpt() {
  local dir="$1"
  local fallback="$2"
  if [[ -f "$fallback" ]]; then
    echo "$fallback"
    return
  fi
  ls -t "$dir"/*.pth 2>/dev/null | head -1
}

ID_CKPT="${ID_CKPT:-$(latest_ckpt ./saved/baseline_beauty_seed${SEED} ./saved/baseline_beauty_seed${SEED}/SASRecAlign-Feb-18-2026_16-14-42.pth)}"
TFIDF_CKPT="${TFIDF_CKPT:-$(latest_ckpt ./saved/two_phase_run_tfidf_v3_stratified ./saved/two_phase_run_tfidf_v3_stratified/SASRecAlignV3-Feb-26-2026_08-32-41.pth)}"
MV_CKPT="${MV_CKPT:-$(latest_ckpt ./saved/peruser_runs/beauty_mv_7b ./saved/phase_runs_multiview_v3_stratified/SASRecAlignMultiViewV3-Feb-26-2026_13-02-36.pth)}"

log "======== Beauty mechanism save_test_scores (seed=$SEED) ========"
log "GPU=$GPU_ID"
log "ID ckpt:     $ID_CKPT"
log "TF-IDF ckpt: $TFIDF_CKPT"
log "MV ckpt:     $MV_CKPT"

log ">>> Eval+scores: ID-only"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_50ep_stratified.yaml" \
  --gpu_id "$GPU_ID" \
  --only_phase_b \
  --phase_b_epochs 0 \
  --resume_from "$ID_CKPT" \
  --seed "$SEED" \
  --save_test_scores \
  --scores_output_dir "$SCORES_DIR" \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR" \
  --variant_features "beauty,id_only,seed${SEED},mechanism" \
  --watchdog_disable \
  >> "$LOG" 2>&1
log "  ✅ ID-only done ($((SECONDS - t0))s)"

log ">>> Eval+scores: TF-IDF V3"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_stratified_v3.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --only_phase_b \
  --phase_b_epochs 0 \
  --resume_from "$TFIDF_CKPT" \
  --seed "$SEED" \
  --save_test_scores \
  --scores_output_dir "$SCORES_DIR" \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR" \
  --variant_features "beauty,tfidf_v3,seed${SEED},mechanism" \
  --watchdog_disable \
  >> "$LOG" 2>&1
log "  ✅ TF-IDF V3 done ($((SECONDS - t0))s)"

log ">>> Eval+scores: MV V3"
t0=$SECONDS
python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_stratified.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --only_phase_b \
  --phase_b_epochs 0 \
  --resume_from "$MV_CKPT" \
  --seed "$SEED" \
  --save_test_scores \
  --scores_output_dir "$SCORES_DIR" \
  --save_peruser_topk \
  --peruser_output_dir "$PERUSER_DIR" \
  --variant_features "beauty,mv_v3,seed${SEED},mechanism" \
  --watchdog_disable \
  >> "$LOG" 2>&1
log "  ✅ MV V3 done ($((SECONDS - t0))s)"

log "======== save_test_scores complete ========"
