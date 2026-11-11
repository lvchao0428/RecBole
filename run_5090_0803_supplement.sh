#!/usr/bin/env bash
#
# 0803 Supplementary Experiments — Serial Queue for 5090
#
# Tasks (in order):
#   1. GTS-Last re-evaluation of all GTS checkpoints (no retraining needed)
#   2. CKA 3-subsample analysis (Beauty + Toys)
#   3. Model cost table (params + inference time)
#
# Expected total runtime: ~60-90 min (mostly eval, no training)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
QLOG="$LOG_DIR/0803_supplement_queue.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

# Source conda env if available
if [ -f scripts/recbole_env.sh ]; then
    source scripts/recbole_env.sh
fi

log "======== 0803 Supplement START (GPU=0) ========"

###############################################################################
# Task 1: GTS-Last re-evaluation
###############################################################################
log "======== [1/3] GTS-Last Re-evaluation ========"

CKPT_LIST="$LOG_DIR/gts_last_ckpt_list.txt"
cat > "$CKPT_LIST" << 'EOF'
# Beauty GTS checkpoints (ID/TF/LLM/MV × 3 seeds)
saved/wrap_beauty_id_lr1e4_do0.1_seed42
saved/wrap_beauty_id_lr1e4_do0.1_seed2024
saved/wrap_beauty_id_lr1e4_do0.1_seed2026
saved/sg_beauty_tf_lr5e4_do0.3_seed42
saved/sg_beauty_tf_lr5e4_do0.3_seed2024
saved/sg_beauty_tf_lr5e4_do0.3_seed2026
saved/sg_beauty_llm_lr1e4_do0.1_seed42
saved/sg_beauty_llm_lr1e4_do0.1_seed2024
saved/sg_beauty_llm_lr1e4_do0.1_seed2026
saved/sg_beauty_mv_lr5e4_do0.3_seed42
saved/sg_beauty_mv_lr5e4_do0.3_seed2024
saved/sg_beauty_mv_lr5e4_do0.3_seed2026
# Beauty UniSRec
saved/unisrec_beauty_gts_seed42
saved/unisrec_beauty_gts_seed2024
saved/unisrec_beauty_gts_seed2026
# Toys GTS checkpoints (ID/TF/LLM/MV × 3 seeds)
saved/wrap_toys_id_lr1e4_do0.1_seed42
saved/wrap_toys_id_lr1e4_do0.1_seed2024
saved/wrap_toys_id_lr1e4_do0.1_seed2026
saved/wrap_toys_tf_lr5e4_do0.3_seed42
saved/wrap_toys_tf_lr5e4_do0.3_seed2024
saved/wrap_toys_tf_lr5e4_do0.3_seed2026
saved/wrap_toys_llm_lr1e4_do0.1_seed42
saved/wrap_toys_llm_lr1e4_do0.1_seed2024
saved/wrap_toys_llm_lr1e4_do0.1_seed2026
saved/wrap_toys_mv_lr5e4_do0.3_seed42
saved/wrap_toys_mv_lr5e4_do0.3_seed2024
saved/wrap_toys_mv_lr5e4_do0.3_seed2026
# Toys UniSRec
saved/unisrec_toys_gts_seed42
saved/unisrec_toys_gts_seed2024
saved/unisrec_toys_gts_seed2026
EOF

GTS_LAST_CSV="$LOG_DIR/gts_last_eval.csv"
log "Running GTS-Last eval on $(grep -c '^saved/' $CKPT_LIST) checkpoints..."
python3 scripts/eval_gts_last.py \
    --ckpt_list "$CKPT_LIST" \
    --out "$GTS_LAST_CSV" \
    --min_train_interactions 5 \
    --gpu_id 0 2>&1 | tee "$LOG_DIR/gts_last_eval.log"

log "GTS-Last eval complete → $GTS_LAST_CSV"

###############################################################################
# Task 2: CKA 3-subsample analysis
###############################################################################
log "======== [2/3] CKA Subsample Analysis ========"

CKA_BEAUTY_JSON="paper_recsys/cka_subsample_beauty.json"
CKA_TOYS_JSON="paper_recsys/cka_subsample_toys.json"

python3 scripts/cka_subsample.py \
    --dataset Amazon_Beauty \
    --output "$CKA_BEAUTY_JSON" 2>&1 | tee "$LOG_DIR/cka_subsample_beauty.log"

python3 scripts/cka_subsample.py \
    --dataset Amazon_Toys_and_Games \
    --output "$CKA_TOYS_JSON" 2>&1 | tee "$LOG_DIR/cka_subsample_toys.log"

log "CKA subsample complete → $CKA_BEAUTY_JSON, $CKA_TOYS_JSON"

###############################################################################
# Task 3: Model cost table
###############################################################################
log "======== [3/3] Model Cost Table ========"

COST_CKPT_LIST="$LOG_DIR/cost_ckpt_list.txt"
cat > "$COST_CKPT_LIST" << 'EOF'
# One checkpoint per model type (Beauty, seed42)
saved/wrap_beauty_id_lr1e4_do0.1_seed42
saved/sg_beauty_tf_lr5e4_do0.3_seed42
saved/sg_beauty_llm_lr1e4_do0.1_seed42
saved/sg_beauty_mv_lr5e4_do0.3_seed42
saved/unisrec_beauty_gts_seed42
# One per Toys
saved/wrap_toys_id_lr1e4_do0.1_seed42
saved/wrap_toys_tf_lr5e4_do0.3_seed42
saved/wrap_toys_llm_lr1e4_do0.1_seed42
saved/wrap_toys_mv_lr5e4_do0.3_seed42
saved/unisrec_toys_gts_seed42
EOF

COST_CSV="$LOG_DIR/model_cost_table.csv"
python3 scripts/model_cost_table.py \
    --ckpt_list "$COST_CKPT_LIST" \
    --out "$COST_CSV" \
    --gpu_id 0 \
    --n_batches 10 2>&1 | tee "$LOG_DIR/model_cost_table.log"

log "Model cost table complete → $COST_CSV"

###############################################################################
# Summary
###############################################################################
log "======== ALL TASKS COMPLETE ========"
log "Artifacts:"
log "  1. GTS-Last eval: $GTS_LAST_CSV"
log "  2. CKA subsample: $CKA_BEAUTY_JSON, $CKA_TOYS_JSON"
log "  3. Cost table: $COST_CSV"
log ""
log "Next: copy results to paper_recsys/ and sync to local"

# Copy key results to paper_recsys for easy access
cp "$GTS_LAST_CSV" paper_recsys/gts_last_eval.csv 2>/dev/null || true
cp "$COST_CSV" paper_recsys/model_cost_table.csv 2>/dev/null || true

log "Done! GPU free."
