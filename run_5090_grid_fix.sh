#!/bin/bash
# Grid Search Fix Script — 接在当前 MV grid 完成后执行
#
# Fix 1: TF-IDF 补跑（文件名加 tfidf_ 前缀，避免被 LLM 覆盖）
# Fix 2: MV 用正确的 two-phase 参数重跑（Phase-A 20 epochs + lr groups + grid）
#
# 执行（当前 grid 完成后）：
#   nohup bash run_5090_grid_fix.sh > logs/grid_fix_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET=Amazon_Beauty
SEED=2025
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU
MIN5=5

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
FIX_LOG="logs/grid_fix_${TIMESTAMP}.log"
FIX_CSV="logs/grid_fix_results_beauty_seed${SEED}.csv"
echo "tag,lr,dropout,wd,valid_mrr,test_mrr" > "$FIX_CSV"

log() { echo "[$(date '+%H:%M')] $*" | tee -a "$FIX_LOG"; }

echo "=== Grid Fix started at $(date) ===" | tee "$FIX_LOG"
echo "Fix 1: TF-IDF 9 runs (tfidf_ prefix)" | tee -a "$FIX_LOG"
echo "Fix 2: MV 9 runs (proper Phase-A: 20 ep + lr groups)" | tee -a "$FIX_LOG"
echo "" | tee -a "$FIX_LOG"

LRS=(0.0001 0.0005 0.001)
DROPOUTS=(0.1 0.3 0.5)
WD=0.0

extract_metrics() {
    local LOG_FILE=$1
    local VALID_MRR=$(grep -oP '"mrr@10": \K[0-9.]+' "$LOG_FILE" | tail -1)
    local TEST_MRR=$(grep -oP '"mrr@10": \K[0-9.]+' "$LOG_FILE" | tail -1)
    echo "${VALID_MRR:-0},${TEST_MRR:-0}"
}

# ============================================================
# Fix 1: TF-IDF — 文件名用 tfidf_ 前缀避免覆盖
# ============================================================
echo "" | tee -a "$FIX_LOG"
echo "=== Fix 1: TF-IDF (tfidf_ prefix, same grid) ===" | tee -a "$FIX_LOG"

for LR in "${LRS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
        TAG="tfidf_lr${LR}_do${DO}"
        LOG_FILE="logs/grid_beauty_${TAG}_seed${SEED}.log"
        CFG_DICT="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':${WD}}"

        log "Running $TAG ..."
        t0=$SECONDS

        python scripts/two_phase_train.py \
            --model "SASRecAlignV3" \
            --dataset "$DATASET" \
            --config_files "$ROOT/sasrec_align_base_stratified_v3_ts.yaml" \
            --config_dict "$CFG_DICT" \
            --gpu_id $GPU \
            --min_train_interactions $MIN5 \
            > "$LOG_FILE" 2>&1 || true

        elapsed=$(( SECONDS - t0 ))
        VALID_MRR=$(grep -oP "mrr@10\s*:\s*\K[0-9.]+" "$LOG_FILE" | tail -1)
        log "$TAG done (${elapsed}s) valid_mrr=${VALID_MRR:-N/A}"
        echo "$TAG,$LR,$DO,$WD,${VALID_MRR:-0}" >> "$FIX_CSV"
    done
done

# ============================================================
# Fix 2: MV — 正确的 Phase-A 参数（对齐 persource_fair 脚本）
# ============================================================
echo "" | tee -a "$FIX_LOG"
echo "=== Fix 2: MV (proper Phase-A: 20ep + lr groups) ===" | tee -a "$FIX_LOG"

for LR in "${LRS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
        TAG="mv_fix_lr${LR}_do${DO}"
        LOG_FILE="logs/grid_beauty_${TAG}_seed${SEED}.log"
        CKPT_DIR="saved/grid_beauty_${TAG}_seed${SEED}"
        CFG_DICT="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':${WD}}"

        log "Running $TAG ..."
        t0=$SECONDS

        python scripts/two_phase_train.py \
            --model "SASRecAlignMultiViewV3" \
            --dataset "$DATASET" \
            --config_files "$ROOT/sasrec_align_multi_view_v3_stratified_ts.yaml" \
            --config_dict "$CFG_DICT" \
            --gpu_id $GPU \
            --min_train_interactions $MIN5 \
            --phase_a_grid --align_grid "0.10" --tau_grid "0.05" \
            --phase_a_epochs 20 --phase_a_eval_step 2 --phase_a_valid_metric "MRR@10" \
            --metric_baseline 0.0 --metric_gain_threshold 0.0 \
            --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
            --phase_a_auto_to_b --phase_b_epochs 40 --backbone_lr_scale 0.1 \
            --checkpoint_dir "$CKPT_DIR" --seed $SEED \
            --save \
            > "$LOG_FILE" 2>&1 || true

        elapsed=$(( SECONDS - t0 ))
        VALID_MRR=$(grep -oP "mrr@10\s*:\s*\K[0-9.]+" "$LOG_FILE" | tail -1)
        log "$TAG done (${elapsed}s) valid_mrr=${VALID_MRR:-N/A}"
        echo "$TAG,$LR,$DO,$WD,${VALID_MRR:-0}" >> "$FIX_CSV"
    done
done

echo "" | tee -a "$FIX_LOG"
echo "=== Grid Fix completed at $(date) ===" | tee -a "$FIX_LOG"
echo "" | tee -a "$FIX_LOG"
echo "Results CSV: $FIX_CSV" | tee -a "$FIX_LOG"

echo "" | tee -a "$FIX_LOG"
echo "=== SUMMARY (sorted by valid_mrr) ===" | tee -a "$FIX_LOG"
sort -t',' -k5 -rn "$FIX_CSV" | head -20 | tee -a "$FIX_LOG"
