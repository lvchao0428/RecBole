#!/usr/bin/env bash
# Early60-EQ [0.6,0.1,0.1] retrain, matched to Beauty/Toys/Grocery main-table protocol.
#
# 0818 failure: this script used the ID-only yaml
#   sasrec_baseline_50ep_stratified_ts_early60_eq.yaml (disable_text_feature: true)
# and did not set disable_text_feature=false in config_dict. Phase-A then had no
# trainable text params (text_mode=none) and crashed with
#   RuntimeError: element 0 of tensors does not require grad
#
# Fix: V3 yamls (text on) + explicit disable_text_feature=false, cutoff-fit
# early60 embeddings, no-cross, boost=0, and a complete eval_args override so
# the split is [0.6,0.1,0.1] (V3 early60 yaml files still say [0.6,0.1,0.3]).
# New checkpoint tags early60eq_ts_v2_* so we do not mix with the crashed run.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
PA_EP=20
PB_EP=50
SEEDS=(42 2024 2026)
DS="Amazon_Beauty"
DS_DIR="$ROOT/dataset/Amazon_Beauty"
LOG="$ROOT/logs/early60eq_fair_0819.log"
CSV="$ROOT/logs/early60eq_fair_0819.csv"
CKPT_LIST="$ROOT/logs/early60eq_fair_ckpts_v2.txt"
TRAIN="$ROOT/scripts/two_phase_train.py"

TF_YAML="sasrec_align_base_stratified_v3_ts_early60.yaml"
LLM_YAML="sasrec_align_qwen3_stratified_v3_ts_early60.yaml"
MV_YAML="sasrec_align_multi_view_v3_stratified_ts_early60.yaml"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

TF_EMB="${DS_DIR}/item_text_emb.base.ts_early60.npy"
LLM_EMB="${DS_DIR}/item_text_emb.qwen2.5_7b.base.ts_early60.npy"
VIEWS_DIR="${DS_DIR}/qwen2.5_7b_4views_ts_early60"
for f in "$TF_EMB" "$LLM_EMB" "$VIEWS_DIR/view_0.npy"; do
  if [[ ! -e "$f" ]]; then
    log "FATAL missing $f"
    exit 1
  fi
done

echo "tag,model,seed,test_mrr_all" > "$CSV"
: > "$CKPT_LIST"

# Full eval_args (not split-only): RecBole replaces the whole eval_args key.
EVAL_ARGS="'eval_args':{'group_by':'user','order':'TO','split':{'TS':[0.6,0.1,0.1]},'mode':{'valid':'full','test':'full'}}"

run_two_phase() {
  local TAG="$1" MODEL="$2" YAML="$3" LR="$4" DO="$5" SEED="$6" EXTRA="$7"
  local CKPT="$ROOT/saved/${TAG}"
  local LOGFILE="$ROOT/logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'cold_threshold':0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0,${EVAL_ARGS}${EXTRA}}"

  log ">>> $TAG"
  python3 "$TRAIN" \
    --model "$MODEL" --dataset "$DS" \
    --config_files "$ROOT/$YAML" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" --min_train_interactions $MIN5 \
    --phase_a_grid --align_grid 0.10 --tau_grid 0.05 \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP --phase_a_eval_step 1 --phase_a_valid_metric MRR@10 \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs $PB_EP --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save --variant_label "$TAG" \
    > "$LOGFILE" 2>&1

  local TEST
  TEST=$(python3 - <<PY
import re, json
text = open("$LOGFILE", errors="replace").read()
val = ""
for s in re.findall(r"\[Run Summary JSON\] (\{.*\})", text):
    try:
        d = json.loads(s)
    except Exception:
        continue
    if d.get("label") != "Phase-B":
        continue
    pb = d.get("results", {}).get("phase_b_result", {})
    val = pb.get("test_result", {}).get("mrr@10", "")
print(val)
PY
)
  echo "${TAG},${MODEL},${SEED},${TEST}" >> "$CSV"
  echo "$CKPT" >> "$CKPT_LIST"
  log "  done $TAG GTS-All test mrr@10=$TEST (paper uses GTS-Last next)"
}

TF_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_path_llm':'','use_llm':false,'text_mode':'base','disable_text_feature':false,'fuse_text_feature':true"
LLM_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_path_llm':'${LLM_EMB}','use_llm':true,'text_mode':'both','disable_text_feature':false,'fuse_text_feature':true"
MV_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_path_llm':'','item_text_emb_split_dir':'${VIEWS_DIR}','use_llm':false,'use_text_view_split':true,'mv_include_base':true,'use_per_view_align':true,'text_view_indices':null,'disable_text_feature':false,'fuse_text_feature':true"

log "======== Early60-EQ fair retrain START (v2) ========"
log "split override TS[0.6,0.1,0.1]; cutoff-fit ts_early60; no-cross; boost=0; text ON"
for SEED in "${SEEDS[@]}"; do
  run_two_phase "early60eq_ts_v2_beauty_tf_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 "$TF_YAML" 0.0005 0.3 "$SEED" "$TF_EXTRA"
  run_two_phase "early60eq_ts_v2_beauty_llm_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 "$LLM_YAML" 0.0001 0.1 "$SEED" "$LLM_EXTRA"
  run_two_phase "early60eq_ts_v2_beauty_mv_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignMultiViewV3 "$MV_YAML" 0.0005 0.3 "$SEED" "$MV_EXTRA"
done

log "======== GTS-Last on fair Early60-EQ v2 ========"
python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$CKPT_LIST" --gpu_id "$GPU_ID" --min_train_interactions 5 \
  --out "$ROOT/paper_recsys/early60eq_gts_last_eval_ts_v2.csv"

log "======== Early60-EQ fair v2 ALL DONE ========"
log "Last: paper_recsys/early60eq_gts_last_eval_ts_v2.csv"
