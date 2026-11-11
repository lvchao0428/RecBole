#!/usr/bin/env bash
# P1 — Beauty main-table TF / LLM / MV on cutoff-fit features.
# Why: sg_beauty_* used YAML default .npy / 4views (not *.ts.npy).
# Does NOT overwrite saved/sg_beauty_*.
# Does NOT retrain Beauty UniSRec (ckpt already records .ts.npy).

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
LOG="$ROOT/logs/beauty_ts_maintable_0818.log"
CSV="$ROOT/logs/beauty_ts_maintable_0818.csv"
CKPT_LIST="$ROOT/logs/beauty_ts_maintable_ckpts.txt"
TRAIN="$ROOT/scripts/two_phase_train.py"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

for f in item_text_emb.base.ts.npy item_text_emb.qwen2.5_7b.base.ts.npy qwen2.5_7b_4views_ts/view_0.npy; do
  if [[ ! -e "$DS_DIR/$f" ]]; then
    log "FATAL missing $DS_DIR/$f"
    exit 1
  fi
done

echo "tag,model,lr,dropout,seed,valid_mrr,test_mrr_all" > "$CSV"
: > "$CKPT_LIST"

run_two_phase() {
  local TAG="$1" MODEL="$2" YAML="$3" LR="$4" DO="$5" SEED="$6" EXTRA="$7"
  local CKPT="$ROOT/saved/${TAG}"
  local LOGFILE="$ROOT/logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'cold_threshold':0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0${EXTRA}}"

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
  echo "${TAG},${MODEL},${LR},${DO},${SEED},,${TEST}" >> "$CSV"
  echo "$CKPT" >> "$CKPT_LIST"
  log "  done GTS-All test mrr@10=$TEST"
}

TF_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','item_text_emb_path_llm':'','use_llm':false,'text_mode':'base'"
LLM_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','item_text_emb_path_llm':'${DS_DIR}/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'text_mode':'both'"
MV_EXTRA=",'item_text_emb_path_base':'${DS_DIR}/item_text_emb.base.ts.npy','item_text_emb_path_llm':'','item_text_emb_split_dir':'${DS_DIR}/qwen2.5_7b_4views_ts','use_llm':false,'use_text_view_split':true,'mv_include_base':true,'use_per_view_align':true,'text_view_indices':null"

log "======== Beauty TS main-table retrain START ========"
for SEED in "${SEEDS[@]}"; do
  run_two_phase "wrap_beauty_tf_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 sasrec_align_base_stratified_v3_ts.yaml 0.0005 0.3 "$SEED" "$TF_EXTRA"
  run_two_phase "wrap_beauty_llm_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 sasrec_align_qwen3_stratified_v3_ts.yaml 0.0001 0.1 "$SEED" "$LLM_EXTRA"
  run_two_phase "wrap_beauty_mv_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignMultiViewV3 sasrec_align_multi_view_v3_stratified_ts.yaml 0.0005 0.3 "$SEED" "$MV_EXTRA"
done

log "======== GTS-Last / Random on Beauty TS ckpts ========"
python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$CKPT_LIST" --gpu_id "$GPU_ID" --min_train_interactions 5 \
  --out "$ROOT/paper_recsys/gts_last_eval_beauty_ts.csv"

for SAMPLE_SEED in 42 123 456; do
  python3 "$ROOT/eval_gts_random.py" \
    --ckpt_list "$CKPT_LIST" --gpu_id "$GPU_ID" --min_train_interactions 5 \
    --sample_seed "$SAMPLE_SEED" \
    --out "$ROOT/paper_recsys/gts_random_eval_beauty_ts_seed${SAMPLE_SEED}.csv"
done

log "======== LOOV GTS-Last on new Beauty MV ========"
LOOV_CSV="$ROOT/paper_recsys/loov_gts_last_beauty_ts.csv"
echo "dataset,seed,config,mrr10" > "$LOOV_CSV"
for SEED in "${SEEDS[@]}"; do
  CKPT="$ROOT/saved/wrap_beauty_mv_lr5e4_do0.3_seed${SEED}"
  OUT="$ROOT/paper_recsys/loov_gts_last_beauty-ts_seed${SEED}.md"
  python3 "$ROOT/scripts/analyze_leave_one_view_out.py" \
    --ckpt "$CKPT" --dataset "$DS" --output "$OUT" \
    --gpu_id "$GPU_ID" --min_train_interactions $MIN5 \
    > "$ROOT/logs/loov_beauty_ts_seed${SEED}.log" 2>&1 || log "WARN LOOV seed=$SEED"
done

log "======== Beauty TS main-table ALL DONE ========"
log "Last: paper_recsys/gts_last_eval_beauty_ts.csv"
