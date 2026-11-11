#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
PA_EP=20
PB_EP=50

LOG="logs/0813_experiments_nohup.log"
mkdir -p logs saved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

extract_mrr() {
  local LOG_FILE=$1
  python3 -c "
import re
t=open('$LOG_FILE',errors='ignore').read()
mv=re.search(r'phase_b_result.*?best_valid_score.: ([0-9.]+)', t, re.S)
mt=re.search(r'phase_b_result.*?\"test_result\": \{.*?\"mrr@10\": ([0-9.]+)', t, re.S)
if not mv:
    mv=re.search(r'best_valid_score.: ([0-9.]+)', t)
if not mt:
    all_m = re.findall(r'\"mrr@10\": ([0-9.]+)', t)
    mt = type('', (), {'group': lambda s,i: all_m[-1]})() if all_m else None
valid = mv.group(1) if mv else '0'
test = mt.group(1) if mt else '0'
print(f'{valid},{test}')
"
}

run_two_phase() {
  local TAG="$1" MODEL="$2" DATASET="$3" YAML="$4" LR="$5" DO="$6" SEED="$7" EXTRA="$8"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0${EXTRA}}"

  log ">>> $TAG (lr=$LR do=$DO seed=$SEED)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --config_files "$ROOT/$YAML" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_grid --align_grid "0.10" --tau_grid "0.05" \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP --phase_a_eval_step 1 --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs $PB_EP --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT" \
    --seed "$SEED" \
    > "$LOGFILE" 2>&1

  local elapsed=$(( SECONDS - t0 ))
  local MRR
  MRR=$(extract_mrr "$LOGFILE")
  log "    done (${elapsed}s) mrr=${MRR}"
  echo "$MRR"
}

# ===========================================================================
# P4 RESUME: Early60-EQ remaining (seed2024 LLM/MV, seed2026 all three)
# ===========================================================================
log "======== RESUME P4: Early60-EQ (remaining) ========"

EARLY_YAML="sasrec_baseline_50ep_stratified_ts_early60_eq.yaml"
TF_EMB="${ROOT}/dataset/Amazon_Beauty/item_text_emb.base.ts_early60.npy"
LLM_EMB="${ROOT}/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts_early60.npy"
VIEWS_DIR="${ROOT}/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts_early60"

TF_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','use_llm':false,'text_mode':'base','disable_text_feature':false,'fuse_text_feature':true"
LLM_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_path_llm':'${LLM_EMB}','use_llm':true,'text_mode':'both','disable_text_feature':false,'fuse_text_feature':true"
MV_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_split_dir':'${VIEWS_DIR}','use_llm':false,'use_text_view_split':true,'mv_include_base':true,'disable_text_feature':false,'fuse_text_feature':true"

CSV="logs/early60eq_results.csv"
echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"
# Record already-done from first run
echo "early60eq_beauty_tf_lr5e4_do0.3_seed42,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,42,0.0237,0.006" >> "$CSV"
echo "early60eq_beauty_llm_lr1e4_do0.1_seed42,SASRecAlignV3,Amazon_Beauty,1e-4,0.1,42,0.0228,0.0058" >> "$CSV"
echo "early60eq_beauty_mv_lr5e4_do0.3_seed42,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,42,0.0237,0.006" >> "$CSV"
echo "early60eq_beauty_tf_lr5e4_do0.3_seed2024,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,2024,0.024,0.0064" >> "$CSV"

# seed2024: LLM and MV
log "-------- Early60-EQ Beauty seed=2024 (resume LLM/MV) --------"
MRR=$(run_two_phase "early60eq_beauty_llm_lr1e4_do0.1_seed2024" SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 1e-4 0.1 2024 "$LLM_EXTRA")
echo "early60eq_beauty_llm_lr1e4_do0.1_seed2024,SASRecAlignV3,Amazon_Beauty,1e-4,0.1,2024,${MRR}" >> "$CSV"

MRR=$(run_two_phase "early60eq_beauty_mv_lr5e4_do0.3_seed2024" SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 5e-4 0.3 2024 "$MV_EXTRA")
echo "early60eq_beauty_mv_lr5e4_do0.3_seed2024,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,2024,${MRR}" >> "$CSV"

# seed2026: all three
log "-------- Early60-EQ Beauty seed=2026 --------"
MRR=$(run_two_phase "early60eq_beauty_tf_lr5e4_do0.3_seed2026" SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 5e-4 0.3 2026 "$TF_EXTRA")
echo "early60eq_beauty_tf_lr5e4_do0.3_seed2026,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,2026,${MRR}" >> "$CSV"

MRR=$(run_two_phase "early60eq_beauty_llm_lr1e4_do0.1_seed2026" SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 1e-4 0.1 2026 "$LLM_EXTRA")
echo "early60eq_beauty_llm_lr1e4_do0.1_seed2026,SASRecAlignV3,Amazon_Beauty,1e-4,0.1,2026,${MRR}" >> "$CSV"

MRR=$(run_two_phase "early60eq_beauty_mv_lr5e4_do0.3_seed2026" SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 5e-4 0.3 2026 "$MV_EXTRA")
echo "early60eq_beauty_mv_lr5e4_do0.3_seed2026,SASRecAlignV3,Amazon_Beauty,5e-4,0.3,2026,${MRR}" >> "$CSV"

log "======== P4 Early60-EQ complete ========"

# ===========================================================================
# P5: Pure LLM-only (Beauty/Toys × 3 seeds)
# ===========================================================================
log "======== P5: Pure LLM-only (no TF-IDF base) ========"

MAIN_YAML="sasrec_baseline_50ep_stratified_ts.yaml"
PURE_LLM_B=",'item_text_emb_path_base':'','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'disable_text_feature':false,'fuse_text_feature':true"
PURE_LLM_T=",'item_text_emb_path_base':'','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'disable_text_feature':false,'fuse_text_feature':true"

CSV5="logs/pure_llm_only.csv"
echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV5"

for SEED in 42 2024 2026; do
  log "-------- Pure-LLM Beauty seed=$SEED --------"
  MRR=$(run_two_phase "purellm_beauty_lr1e4_do0.1_seed${SEED}" SASRecAlignV3 Amazon_Beauty "$MAIN_YAML" 1e-4 0.1 "$SEED" "$PURE_LLM_B")
  echo "purellm_beauty_lr1e4_do0.1_seed${SEED},SASRecAlignV3,Amazon_Beauty,1e-4,0.1,${SEED},${MRR}" >> "$CSV5"

  log "-------- Pure-LLM Toys seed=$SEED --------"
  MRR=$(run_two_phase "purellm_toys_lr1e4_do0.1_seed${SEED}" SASRecAlignV3 Amazon_Toys_and_Games "$MAIN_YAML" 1e-4 0.1 "$SEED" "$PURE_LLM_T")
  echo "purellm_toys_lr1e4_do0.1_seed${SEED},SASRecAlignV3,Amazon_Toys_and_Games,1e-4,0.1,${SEED},${MRR}" >> "$CSV5"
done

log "======== P5 Pure LLM-only complete ========"
log "======== ALL EXPERIMENTS DONE ========"
# After training completes, run raw 4096-d CKA audit:
bash scripts/generate_raw_4096d_for_cka.sh
