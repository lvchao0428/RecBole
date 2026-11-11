#!/usr/bin/env bash
# 0813 Advisor guidance — three experiment blocks (serial):
#   P2) LOOV under GTS-Last (3 seeds × Beauty/Toys, zero-mask, no retraining)
#   P4) Early [0.6,0.1,0.1] equal-horizon rolling window (TF/LLM/MV × 3 seeds × Beauty)
#   P5) TF+LLM Hybrid (Beauty/Toys × 3 seeds)
#
# Total GPU time: ~15h
# Usage:
#   nohup bash run_5090_0813_experiments.sh > logs/0813_experiments.log 2>&1 &

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
SEEDS=(42 2024 2026)

LOG="logs/0813_experiments.log"
mkdir -p logs saved paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

extract_mrr() {
  local LOG_FILE=$1
  python3 -c "
import re
t=open('$LOG_FILE',errors='ignore').read()
# Try phase_b_result first
mv=re.search(r'phase_b_result.*?best_valid_score.: ([0-9.]+)', t, re.S)
mt=re.search(r'phase_b_result.*?\"test_result\": \{.*?\"mrr@10\": ([0-9.]+)', t, re.S)
if not mv:
    mv=re.search(r'best_valid_score.: ([0-9.]+)', t)
if not mt:
    # Last mrr@10 in test result
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
  echo "${TAG},SASRecAlignV3,${DATASET},${LR},${DO},${SEED},${MRR}" >> "$CSV"
}

# ===========================================================================
# P2: LOOV under GTS-Last (zero-mask, no retraining)
# ===========================================================================
log "======== P2: LOOV under GTS-Last (3 seeds × Beauty/Toys) ========"

LOOV_CSV="paper_recsys/loov_gts_last.csv"
echo "dataset,seed,config,mrr10,delta_desc,delta_func,delta_aud,delta_style" > "$LOOV_CSV"

for DS in Amazon_Beauty Amazon_Toys_and_Games; do
  DS_SHORT=$(echo "$DS" | sed 's/Amazon_//' | tr 'A-Z' 'a-z' | tr '_' '-')
  for SEED in "${SEEDS[@]}"; do
    if [ "$DS" = "Amazon_Beauty" ]; then
      CKPT="saved/sg_beauty_mv_lr5e4_do0.3_seed${SEED}"
    else
      CKPT="saved/wrap_toys_mv_lr5e4_do0.3_seed${SEED}"
    fi

    if [ ! -d "$CKPT" ]; then
      log "  SKIP $CKPT (not found)"
      continue
    fi

    OUTFILE="paper_recsys/loov_gts_last_${DS_SHORT}_seed${SEED}.md"
    log "  LOOV $DS seed=$SEED -> $OUTFILE"
    python scripts/analyze_leave_one_view_out.py \
      --ckpt "$CKPT" \
      --dataset "$DS" \
      --output "$OUTFILE" \
      --gpu_id "$GPU_ID" \
      --min_train_interactions $MIN5 \
      > "logs/loov_gts_last_${DS_SHORT}_seed${SEED}.log" 2>&1

    # Extract numbers for CSV
    python3 -c "
import re
t = open('$OUTFILE').read()
# Find full MRR line
full_m = re.search(r'Full.*?\*\*([0-9.]+)\*\*', t)
full_mrr = full_m.group(1) if full_m else '0'
# Find delta lines (Drop view_X)
deltas = re.findall(r'Drop view_\d.*?([+-][0-9.]+)\s*\|', t)
while len(deltas) < 4:
    deltas.append('0')
print(f'$DS,$SEED,gts_last_warm,{full_mrr},{deltas[0]},{deltas[1]},{deltas[2]},{deltas[3]}')
" >> "$LOOV_CSV"
    log "    done"
  done
done

log "  LOOV results: $LOOV_CSV"

# ===========================================================================
# P4: Early [0.6,0.1,0.1] equal-horizon (TF/LLM/MV × 3 seeds × Beauty)
# ===========================================================================
log "======== P4: Early60 Equal-Horizon [0.6,0.1,0.1] Beauty ========"

CSV="logs/early60_eq_beauty.csv"
echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"

EARLY_YAML="sasrec_baseline_50ep_stratified_ts_early60_eq.yaml"

# Use the existing TS-early60 features (they were built for 60% cutoff)
TF_EMB="${ROOT}/dataset/Amazon_Beauty/item_text_emb.base.ts_early60.npy"
LLM_EMB="${ROOT}/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts_early60.npy"
VIEWS_DIR="${ROOT}/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts_early60"

TF_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','use_llm':false,'text_mode':'base','disable_text_feature':false,'fuse_text_feature':true"
LLM_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_path_llm':'${LLM_EMB}','use_llm':true,'text_mode':'both','disable_text_feature':false,'fuse_text_feature':true"
MV_EXTRA=",'item_text_emb_path_base':'${TF_EMB}','item_text_emb_split_dir':'${VIEWS_DIR}','use_llm':false,'use_text_view_split':true,'mv_include_base':true,'disable_text_feature':false,'fuse_text_feature':true"

for SEED in "${SEEDS[@]}"; do
  log "-------- Early60-EQ Beauty seed=$SEED --------"

  run_two_phase "early60eq_beauty_tf_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 5e-4 0.3 "$SEED" "$TF_EXTRA"

  run_two_phase "early60eq_beauty_llm_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 1e-4 0.1 "$SEED" "$LLM_EXTRA"

  run_two_phase "early60eq_beauty_mv_lr5e4_do0.3_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$EARLY_YAML" 5e-4 0.3 "$SEED" "$MV_EXTRA"
done

log "  Early60-EQ results: $CSV"

# ===========================================================================
# P5: Pure LLM-only (no TF-IDF base) — to prove TF+LLM hybrid value
# Current "LLM" column = TF+LLM hybrid. Need pure LLM-only for comparison.
# ===========================================================================
log "======== P5: Pure LLM-only (Beauty/Toys × 3 seeds) ========"

CSV="logs/pure_llm_only.csv"
echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"

# Pure LLM: set item_text_emb_path_base to empty/none, only use item_text_emb_path_llm
# By setting base to "" and only providing llm path, text_mode should become "llm"
PURE_LLM_B=",'item_text_emb_path_base':'','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'disable_text_feature':false,'fuse_text_feature':true"
PURE_LLM_T=",'item_text_emb_path_base':'','item_text_emb_path_llm':'${ROOT}/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy','use_llm':true,'disable_text_feature':false,'fuse_text_feature':true"

for SEED in "${SEEDS[@]}"; do
  log "-------- Pure-LLM Beauty seed=$SEED --------"
  run_two_phase "purellm_beauty_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 Amazon_Beauty "$MAIN_YAML_B" 1e-4 0.1 "$SEED" "$PURE_LLM_B"

  log "-------- Pure-LLM Toys seed=$SEED --------"
  run_two_phase "purellm_toys_lr1e4_do0.1_seed${SEED}" \
    SASRecAlignV3 Amazon_Toys_and_Games "$MAIN_YAML_T" 1e-4 0.1 "$SEED" "$PURE_LLM_T"
done

log "  Pure LLM-only results: $CSV"

# ===========================================================================
# Post-hoc: GTS-Last eval on early60eq and hybrid checkpoints
# ===========================================================================
log "======== Post-hoc: GTS-Last/Random eval on new checkpoints ========"

# Build checkpoint list for GTS-Last eval
CKPT_FILE="/tmp/ckpt_list_0813.txt"
: > "$CKPT_FILE"
for SEED in "${SEEDS[@]}"; do
  echo "saved/early60eq_beauty_tf_lr5e4_do0.3_seed${SEED}" >> "$CKPT_FILE"
  echo "saved/early60eq_beauty_llm_lr1e4_do0.1_seed${SEED}" >> "$CKPT_FILE"
  echo "saved/early60eq_beauty_mv_lr5e4_do0.3_seed${SEED}" >> "$CKPT_FILE"
  echo "saved/purellm_beauty_lr1e4_do0.1_seed${SEED}" >> "$CKPT_FILE"
  echo "saved/purellm_toys_lr1e4_do0.1_seed${SEED}" >> "$CKPT_FILE"
done

# GTS-Last
log "  GTS-Last eval on new checkpoints..."
python3 eval_gts_last.py \
  --ckpt_list "$CKPT_FILE" \
  --out logs/gts_last_eval_0813.csv \
  --min_train_interactions $MIN5 \
  --gpu_id "$GPU_ID" 2>&1 | tee -a "$LOG"

# GTS-Random (3 sample seeds)
for SSEED in 42 123 456; do
  log "  GTS-Random seed=$SSEED on new checkpoints..."
  python3 eval_gts_random.py \
    --ckpt_list "$CKPT_FILE" \
    --out "logs/gts_random_eval_0813_seed${SSEED}.csv" \
    --sample_seed "$SSEED" \
    --min_train_interactions $MIN5 \
    --gpu_id "$GPU_ID" 2>&1 | tee -a "$LOG"
done

# ===========================================================================
log "======== ALL 0813 EXPERIMENTS COMPLETE ========"
log "Artifacts:"
log "  LOOV GTS-Last: paper_recsys/loov_gts_last.csv + per-seed .md files"
log "  Early60-EQ: logs/early60_eq_beauty.csv"
log "  Hybrid: logs/hybrid_tfidf_llm.csv"
log "  GTS-Last new: logs/gts_last_eval_0813.csv"
log "  GTS-Random new: logs/gts_random_eval_0813_seed{42,123,456}.csv"
