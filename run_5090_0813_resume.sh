#!/bin/bash
set -e
source scripts/recbole_env.sh
export PYTHONPATH=/home/charlie/project/RecBole:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

ROOT=/home/charlie/project/RecBole
LOG=$ROOT/logs/0813_experiments_nohup.log
CSV=$ROOT/logs/early60eq_results.csv

echo "[$(date +%F\ %H:%M:%S)] ======== RESUME from early60eq_beauty_llm seed2024 ========" >> $LOG

# Initialize CSV if needed
if [ ! -f "$CSV" ]; then
  echo "dataset,seed,model,mrr_phase_a,mrr_phase_b" > $CSV
fi

# Append the already-done results from seed42 and seed2024-TF (from nohup log)
# seed42: tf=0.0237,0.006  llm=0.0228,0.0058  mv=0.0237,0.006
# seed2024: tf=0.024,0.0064
grep -q "early60eq_beauty_tf.*seed42" $CSV 2>/dev/null || echo "Amazon_Beauty,42,tf,0.0237,0.006" >> $CSV
grep -q "early60eq_beauty_llm.*seed42" $CSV 2>/dev/null || echo "Amazon_Beauty,42,llm,0.0228,0.0058" >> $CSV
grep -q "early60eq_beauty_mv.*seed42" $CSV 2>/dev/null || echo "Amazon_Beauty,42,mv,0.0237,0.006" >> $CSV
grep -q "early60eq_beauty_tf.*seed2024" $CSV 2>/dev/null || echo "Amazon_Beauty,2024,tf,0.024,0.0064" >> $CSV

run_two_phase() {
  local TAG=$1 LR=$2 DO=$3 SEED=$4 EMB_BASE=$5 EMB_LLM=$6 VIEWS_DIR=$7
  local LOGF=$ROOT/logs/${TAG}.log
  local START=$(date +%s)
  echo "[$(date +%F\ %H:%M:%S)] >>> $TAG (lr=$LR do=$DO seed=$SEED)" >> $LOG

  python3 scripts/two_phase_train.py \
    --config_file sasrec_baseline_50ep_stratified_ts_early60_eq.yaml \
    --seed $SEED --learning_rate $LR --hidden_dropout_prob $DO \
    --item_text_emb_path_base "$EMB_BASE" \
    --item_text_emb_path_llm "$EMB_LLM" \
    ${VIEWS_DIR:+--views_dir "$VIEWS_DIR"} \
    --tag "$TAG" --gpu_id 0 > "$LOGF" 2>&1

  local END=$(date +%s)
  local DUR=$((END-START))
  local MRR=$(grep -oP "mrr@10\s*:\s*\K[0-9.]+" "$LOGF" | tail -2 | paste -sd,)
  echo "[$(date +%F\ %H:%M:%S)]     done (${DUR}s) mrr=$MRR" >> $LOG
  echo "$MRR"
}

TFIDF_E60="$ROOT/dataset/Amazon_Beauty/item_text_emb.tfidf_svd.stratified.ts.early60_eq.npy"
QWEN_E60="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.early60_eq.npy"
VIEWS_E60="$ROOT/dataset/Amazon_Beauty/views_ts_early60_eq"

# --- seed2024: LLM and MV remaining ---
MRR=$(run_two_phase "early60eq_beauty_llm_lr1e4_do0.1_seed2024" 1e-4 0.1 2024 "" "$QWEN_E60" "")
echo "Amazon_Beauty,2024,llm,${MRR}" >> $CSV

MRR=$(run_two_phase "early60eq_beauty_mv_lr5e4_do0.3_seed2024" 5e-4 0.3 2024 "" "" "$VIEWS_E60")
echo "Amazon_Beauty,2024,mv,${MRR}" >> $CSV

# --- seed2026: all three ---
for SEED in 2026; do
  MRR=$(run_two_phase "early60eq_beauty_tf_lr5e4_do0.3_seed${SEED}" 5e-4 0.3 $SEED "$TFIDF_E60" "" "")
  echo "Amazon_Beauty,${SEED},tf,${MRR}" >> $CSV

  MRR=$(run_two_phase "early60eq_beauty_llm_lr1e4_do0.1_seed${SEED}" 1e-4 0.1 $SEED "" "$QWEN_E60" "")
  echo "Amazon_Beauty,${SEED},llm,${MRR}" >> $CSV

  MRR=$(run_two_phase "early60eq_beauty_mv_lr5e4_do0.3_seed${SEED}" 5e-4 0.3 $SEED "" "" "$VIEWS_E60")
  echo "Amazon_Beauty,${SEED},mv,${MRR}" >> $CSV
done

echo "[$(date +%F\ %H:%M:%S)] ======== P4 Early60-EQ done ========" >> $LOG

# --- P5: Pure LLM-only (Beauty/Toys × 3 seeds) ---
echo "[$(date +%F\ %H:%M:%S)] ======== P5: Pure LLM-only (no TF-IDF base) ========" >> $LOG
CSV5=$ROOT/logs/purellm_results.csv
echo "dataset,seed,mrr_phase_a,mrr_phase_b" > $CSV5

QWEN_BEAUTY="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
QWEN_TOYS="$ROOT/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy"

for SEED in 42 2024 2026; do
  MRR=$(run_two_phase "purellm_beauty_lr1e4_do0.1_seed${SEED}" 1e-4 0.1 $SEED "" "$QWEN_BEAUTY" "")
  echo "Amazon_Beauty,${SEED},${MRR}" >> $CSV5

  MRR=$(run_two_phase "purellm_toys_lr1e4_do0.1_seed${SEED}" 1e-4 0.1 $SEED "" "$QWEN_TOYS" "")
  echo "Amazon_Toys_and_Games,${SEED},${MRR}" >> $CSV5
done

echo "[$(date +%F\ %H:%M:%S)] ======== P5 Pure LLM-only done ========" >> $LOG
echo "[$(date +%F\ %H:%M:%S)] ======== ALL DONE ========" >> $LOG
