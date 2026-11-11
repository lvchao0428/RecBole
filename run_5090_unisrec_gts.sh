#!/usr/bin/env bash
# UniSRec GTS baseline — Beauty + Toys, 3 seeds
# Using unified protocol: TS[0.8,0.1,0.1], min_train_interactions=5
#
# Usage:
#   nohup bash run_5090_unisrec_gts.sh > logs/unisrec_gts.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
SEEDS=(42 2024 2026)
CSV="logs/unisrec_gts_results.csv"
QLOG="logs/unisrec_gts_queue.log"
mkdir -p logs

echo "tag,model,dataset,seed,valid_mrr,test_mrr" > "$CSV"
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

extract_mrr() {
  local LOG_FILE=$1
  python3 -c "
import re, sys
t = open(sys.argv[1], errors='ignore').read()
mrr_vals = re.findall(r'mrr@10 : ([0-9.]+)', t)
valid = '0'
m = re.search(r'best_valid_score.: ([0-9.]+)', t)
if m: valid = m.group(1)
test = mrr_vals[-1] if mrr_vals else '0'
print(f'{valid},{test}')
" "$LOG_FILE"
}

run_unisrec() {
  local TAG="$1" DATASET="$2" YAML="$3" SEED="$4"
  local LOGFILE="logs/${TAG}.log"
  local CKPT="./saved/${TAG}"

  log ">>> $TAG"
  rm -rf "$CKPT"
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
  local METRICS VALID TEST
  METRICS=$(extract_mrr "$LOGFILE")
  VALID=${METRICS%,*}
  TEST=${METRICS#*,}
  log "  done (${elapsed}s) valid=${VALID} test=${TEST}"
  echo "${TAG},UniSRec,${DATASET},${SEED},${VALID},${TEST}" >> "$CSV"
}

log "======== UniSRec GTS Baseline START ========"

# Beauty GTS
BEAUTY_YAML="unisrec_beauty_stratified_ts.yaml"
for SEED in "${SEEDS[@]}"; do
  run_unisrec "unisrec_beauty_gts_seed${SEED}" Amazon_Beauty "$BEAUTY_YAML" "$SEED"
done

# Toys GTS (need to create config if not exists)
TOYS_YAML="unisrec_toys_stratified_ts.yaml"
if [[ ! -f "$TOYS_YAML" ]]; then
  log "Creating $TOYS_YAML from beauty config..."
  sed -e 's/Amazon_Beauty/Amazon_Toys_and_Games/g' \
      -e 's|dataset/Amazon_Beauty|dataset/Amazon_Toys_and_Games|g' \
      "$BEAUTY_YAML" > "$TOYS_YAML"
fi

for SEED in "${SEEDS[@]}"; do
  run_unisrec "unisrec_toys_gts_seed${SEED}" Amazon_Toys_and_Games "$TOYS_YAML" "$SEED"
done

log "======== UniSRec GTS Baseline COMPLETE ========"
