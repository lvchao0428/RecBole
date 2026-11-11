#!/usr/bin/env bash
# WSDM 0723 wrap-up — log10 (1080 Ti 11GB) serial queue
#
# Suitable for log10 (lighter / analysis):
#   1) Warm-target audit + distribution shift + view redundancy (CPU/light)
#   2) Beauty ID-only × seeds {42,2024,2026}
#   3) UniSRec Beauty seed=2025 (portability; existing script pattern)
#
# Usage (log10):
#   nohup bash run_log10_wsdm_wrapup_serial.sh > logs/wsdm_wrapup_log10_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
SEEDS=(42 2024 2026)

CSV="logs/wsdm_wrapup_log10.csv"
QLOG="logs/wsdm_wrapup_log10_queue.log"
mkdir -p logs saved paper_recsys

echo "tag,model,dataset,lr,dropout,seed,valid_mrr,test_mrr" > "$CSV"
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

log "======== WSDM 0723 wrap-up (log10) started ========"

# Kill stale waiters from old queues (safe: only the 7/11 wait loops)
pkill -f 'while pgrep -f two_phase_train' 2>/dev/null || true

# -------- G1 / G3 analysis (CPU) --------
log "======== Analysis: warm-target + shift + view redundancy ========"
python scripts/audit_warm_targets.py --dataset Amazon_Beauty \
  --out paper_recsys/warm_target_audit_amazon_beauty.md || true
python scripts/analyze_distribution_shift.py --dataset Amazon_Beauty || true
if [[ -f scripts/analyze_view_redundancy.py ]]; then
  python scripts/analyze_view_redundancy.py --dataset Amazon_Beauty || true
fi
if [[ -d dataset/Amazon_Toys_and_Games ]]; then
  python scripts/audit_warm_targets.py --dataset Amazon_Toys_and_Games \
    --out paper_recsys/warm_target_audit_amazon_toys_and_games.md || true
fi

# -------- Beauty ID-only 3-seed --------
log "======== Beauty ID-only 3-seed ========"
for SEED in "${SEEDS[@]}"; do
  TAG="wrap_beauty_id_lr1e4_do0.1_seed${SEED}"
  LOGFILE="logs/${TAG}.log"
  CKPT="./saved/${TAG}"
  CFG="{'learning_rate':0.0001,'attn_dropout_prob':0.1,'hidden_dropout_prob':0.1,'weight_decay':0.0,'freeze_backbone':false,'disable_text_feature':true,'fuse_text_feature':false,'cold_text_boost':0.0,'infer_boost':0.0}"
  log ">>> $TAG"
  t0=$SECONDS
  python scripts/two_phase_train.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files "$ROOT/sasrec_baseline_50ep_stratified_ts.yaml" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_epochs 50 --phase_a_eval_step 5 --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1 || true
  VALID=$(grep -oP 'best_valid_score.: \K[0-9.]+' "$LOGFILE" | tail -1 || echo 0)
  TEST=$(python3 -c "import re;t=open('$LOGFILE',errors='ignore').read();m=re.findall(r'\"mrr@10\": ([0-9.]+)',t);print(m[-1] if m else '0')")
  log "  done ($((SECONDS-t0))s) valid=${VALID:-0} test=${TEST:-0}"
  echo "${TAG},SASRecAlign,Amazon_Beauty,0.0001,0.1,${SEED},${VALID:-0},${TEST:-0}" >> "$CSV"
done

# -------- UniSRec Beauty seed2025 (if script present) --------
if [[ -f "$ROOT/run_log10_ts_unisrec_seed2025.sh" ]]; then
  log "======== UniSRec Beauty seed2025 ========"
  bash "$ROOT/run_log10_ts_unisrec_seed2025.sh" || true
else
  log "SKIP UniSRec: run_log10_ts_unisrec_seed2025.sh missing"
fi

# Sync analysis docs back to 5090 if reachable
if ping -c1 -W1 192.168.0.106 >/dev/null 2>&1 || true; then
  log "Syncing paper_recsys audits → 5090 (best-effort)"
  rsync -avz paper_recsys/warm_target_audit_*.md paper_recsys/warm_target_audit_*.json \
    charlie@192.168.0.106:/home/charlie/project/RecBole/paper_recsys/ 2>/dev/null || true
  rsync -avz logs/wsdm_wrapup_log10.csv logs/wsdm_wrapup_log10_queue.log \
    charlie@192.168.0.106:/home/charlie/project/RecBole/logs/ 2>/dev/null || true
fi

log "======== WSDM wrap-up log10 COMPLETE ========"
touch logs/wsdm_wrapup_log10.DONE
cat "$CSV" | tee -a "$QLOG"
