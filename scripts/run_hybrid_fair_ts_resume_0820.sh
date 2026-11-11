#!/usr/bin/env bash
# Resume the matched-protocol TF + single-LLM-view hybrid.
# hybrid_ts_beauty_seed42 finished 2026-08-19 11:19 and is skipped.
# hybrid_ts_toys_seed42 has Phase-A only and is retrained from scratch.
#
# Protocol identical to run_hybrid_fair_ts_0818.sh: cutoff-fit *.ts.npy +
# qwen2.5_7b_4views_ts, 1 view (Description), no-cross, no-boost, min5,
# MV hyperparams (lr 5e-4 / dropout 0.3), Phase-A 20 / Phase-B 50.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MV_YAML="$ROOT/sasrec_align_multi_view_v3_stratified_ts.yaml"
LOG="$ROOT/logs/hybrid_fair_ts_resume_0820.log"
TRAIN="$ROOT/scripts/two_phase_train.py"
: > "$LOG"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

phase_b_done() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ $(ls -1 "$d"/*.pth 2>/dev/null | wc -l) -ge 2 ]] || return 1
  grep -q "Phase-B-post-free" "$ROOT/logs/$(basename "$d").log" 2>/dev/null
}

run_one() {
  local DATASET=$1 SEED=$2 SHORT=$3
  local TAG="hybrid_ts_${SHORT}_seed${SEED}"
  local DS_DIR="$ROOT/dataset/${DATASET}"
  local BASE="${DS_DIR}/item_text_emb.base.ts.npy"
  local VIEWS="${DS_DIR}/qwen2.5_7b_4views_ts"
  local CKPT_DIR="$ROOT/saved/${TAG}"

  if [[ ! -f "$BASE" || ! -d "$VIEWS" ]]; then
    log "FATAL missing cutoff-fit features for $DATASET"
    exit 1
  fi

  if phase_b_done "$CKPT_DIR"; then
    log "SKIP $TAG (Phase-B already complete)"
    return 0
  fi

  rm -rf "$CKPT_DIR"
  log "--- train $TAG ---"
  python3 "$TRAIN" \
    --model SASRecAlignMultiViewV3 --dataset "$DATASET" \
    --config_files "$MV_YAML" \
    --config_dict "{'learning_rate':5e-4,'attn_dropout_prob':0.3,'hidden_dropout_prob':0.3,'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'cold_threshold':0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0,'use_text_view_split':true,'mv_include_base':true,'use_per_view_align':true,'num_text_views':1,'text_view_indices':[0],'item_text_emb_path_base':'${BASE}','item_text_emb_path_llm':'','item_text_emb_split_dir':'${VIEWS}','use_llm':false,'disable_text_feature':false,'fuse_text_feature':true}" \
    --gpu_id 0 --min_train_interactions 5 \
    --phase_a_grid --align_grid 0.10 --tau_grid 0.05 \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs 20 --phase_a_eval_step 1 \
    --phase_a_valid_metric MRR@10 \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs 50 \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT_DIR" \
    --seed "$SEED" --save --variant_label "$TAG" \
    > "$ROOT/logs/${TAG}.log" 2>&1
  log "  done $TAG"
}

log "======== hybrid resume START ========"
for SEED in 2024 2026; do
  run_one Amazon_Beauty "$SEED" beauty
done
for SEED in 42 2024 2026; do
  run_one Amazon_Toys_and_Games "$SEED" toys
done
log "======== train done -> GTS-Last / Random ========"

LIST="$ROOT/logs/hybrid_ts_ckpts_0820.txt"
: > "$LIST"
for SEED in 42 2024 2026; do
  echo "$ROOT/saved/hybrid_ts_beauty_seed${SEED}" >> "$LIST"
  echo "$ROOT/saved/hybrid_ts_toys_seed${SEED}" >> "$LIST"
done

python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$LIST" --gpu_id 0 --min_train_interactions 5 \
  --out "$ROOT/paper_recsys/hybrid_ts_gts_last.csv" >> "$LOG" 2>&1
log "  Last -> paper_recsys/hybrid_ts_gts_last.csv"

for SAMPLE_SEED in 42 123 456; do
  python3 "$ROOT/scripts/eval_gts_random.py" \
    --ckpt_list "$LIST" --gpu_id 0 --min_train_interactions 5 \
    --sample_seed "$SAMPLE_SEED" \
    --out "$ROOT/paper_recsys/hybrid_ts_gts_random_seed${SAMPLE_SEED}.csv" >> "$LOG" 2>&1
  log "  Random seed=${SAMPLE_SEED} done"
done

touch "$ROOT/logs/hybrid_fair_ts_resume_0820.DONE"
log "======== hybrid resume ALL DONE ========"
