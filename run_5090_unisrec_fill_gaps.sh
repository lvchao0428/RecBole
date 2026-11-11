#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
QLOG="logs/unisrec_fill_gaps.log"
CSV_LOO="paper_recsys/unisrec_loo_results.csv"
CSV_GROCERY="paper_recsys/unisrec_grocery_gts_results.csv"
mkdir -p logs saved paper_recsys
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

extract_metrics() {
  python -c "
import re, sys
t = open(sys.argv[1], errors='ignore').read()
mrr  = re.findall(r'mrr@10 : ([0-9.]+)', t)
ndcg = re.findall(r'ndcg@10 : ([0-9.]+)', t)
hit  = re.findall(r'hit@10 : ([0-9.]+)', t)
m = re.search(r'best_valid_score.: ([0-9.]+)', t)
valid = m.group(1) if m else '0'
print(f'{valid},{mrr[-1] if mrr else 0},{ndcg[-1] if ndcg else 0},{hit[-1] if hit else 0}')
" "$1"
}

run_unisrec() {
  local TAG="$1" DATASET="$2" YAML="$3" SEED="$4" CSV_OUT="$5"
  local LOGFILE="logs/${TAG}.log"
  local CKPT="./saved/${TAG}"

  if [[ -d "$CKPT" ]]; then
    log "SKIP $TAG (exists)"
    return
  fi

  log "Training $TAG ..."
  local t0=$SECONDS

  python scripts/two_phase_train.py \
    --model UniSRec \
    --dataset "$DATASET" \
    --config_files "$ROOT/$YAML" \
    --config_dict "{'seed':${SEED}}" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions 5 \
    --phase_a_epochs 50 --phase_a_eval_step 5 --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1 || true

  local elapsed=$(( SECONDS - t0 ))
  local METRICS
  METRICS=$(extract_metrics "$LOGFILE")
  IFS=',' read -r VALID TEST_MRR TEST_NDCG TEST_HIT <<< "$METRICS"
  log "  done (${elapsed}s) valid=${VALID} mrr=${TEST_MRR} ndcg=${TEST_NDCG} hit=${TEST_HIT}"
  echo "${TAG},UniSRec,${DATASET},${SEED},${VALID},${TEST_MRR},${TEST_NDCG},${TEST_HIT}" >> "$CSV_OUT"
}

log "======== UniSRec gap-fill START (GPU=$GPU_ID) ========"

# ---- Task 1: UniSRec LOO Beauty + Toys (3 seeds each) ----
echo "tag,model,dataset,seed,valid_mrr,test_mrr,test_ndcg,test_hit" > "$CSV_LOO"

for SEED in 42 2024 2026; do
  run_unisrec "unisrec_loo_beauty_seed${SEED}" Amazon_Beauty \
    unisrec_beauty_loo.yaml "$SEED" "$CSV_LOO"
done

for SEED in 42 2024 2026; do
  run_unisrec "unisrec_loo_toys_seed${SEED}" Amazon_Toys_and_Games \
    unisrec_toys_loo.yaml "$SEED" "$CSV_LOO"
done

log "---- LOO results ----"
cat "$CSV_LOO" | tee -a "$QLOG"

# ---- Task 2: UniSRec GTS Grocery (3 seeds) ----
echo "tag,model,dataset,seed,valid_mrr,test_mrr,test_ndcg,test_hit" > "$CSV_GROCERY"

for SEED in 42 2024 2026; do
  run_unisrec "unisrec_grocery_gts_seed${SEED}" Amazon_Grocery_and_Gourmet_Food \
    unisrec_grocery_stratified_ts.yaml "$SEED" "$CSV_GROCERY"
done

log "---- Grocery GTS results ----"
cat "$CSV_GROCERY" | tee -a "$QLOG"

touch logs/unisrec_fill_gaps.DONE
log "======== UniSRec gap-fill COMPLETE ========"
