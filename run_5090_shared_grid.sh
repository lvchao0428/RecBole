#!/bin/bash
# WSDM V4 Shared Small Grid Search - Beauty
# 按 0717 老师建议：共享小网格，lr/dropout 各 3 值，同预算
# 先跑精简 9 组（lr×3, dropout×3, wd=0），每模型 ~3h
# 总计 3 models × 9 configs = 27 runs，预计 9-12h
#
# 执行：nohup bash run_5090_shared_grid.sh > logs/shared_grid_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON=/home/charlie/anaconda3/bin/python
DATASET=Amazon_Beauty
SEED=2025
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
GRID_LOG="logs/shared_grid_${TIMESTAMP}.log"
echo "=== Shared Grid Search started at $(date) ===" | tee $GRID_LOG
echo "Grid: lr=[1e-4, 5e-4, 1e-3] × dropout=[0.1, 0.3, 0.5] × wd=0" | tee -a $GRID_LOG
echo "Models: TF-IDF, LLM, MV" | tee -a $GRID_LOG
echo "" | tee -a $GRID_LOG

# Grid values
LRS=(0.0001 0.0005 0.001)
DROPOUTS=(0.1 0.3 0.5)
WD=0.0

MIN5=5

run_experiment() {
    local MODEL=$1
    local CONFIG=$2
    local LR=$3
    local DROPOUT=$4
    local TAG="${MODEL}_lr${LR}_do${DROPOUT}"
    
    echo "[$(date '+%H:%M')] Running $TAG ..." | tee -a $GRID_LOG
    
    local LOG_FILE="logs/grid_beauty_${TAG}_seed${SEED}.log"
    
    local CFG_DICT="{'learning_rate':${LR},'attn_dropout_prob':${DROPOUT},'hidden_dropout_prob':${DROPOUT},'weight_decay':${WD}}"
    
    python scripts/two_phase_train.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --config_files "$CONFIG" \
        --config_dict "$CFG_DICT" \
        --gpu_id $GPU \
        --min_train_interactions $MIN5 \
        > $LOG_FILE 2>&1 || true
    
    local EXIT_CODE=$?
    
    # Extract valid best MRR
    local VALID_MRR=$(grep -oP "mrr@10\s*:\s*\K[0-9.]+" $LOG_FILE | tail -1)
    
    echo "[$(date '+%H:%M')] $TAG done (exit=$EXIT_CODE) valid_mrr=${VALID_MRR:-N/A}" | tee -a $GRID_LOG
    echo "$TAG,$LR,$DROPOUT,$WD,${VALID_MRR:-0}" >> "logs/grid_results_beauty_seed${SEED}.csv"
}

# Initialize CSV
echo "tag,lr,dropout,wd,valid_mrr" > "logs/grid_results_beauty_seed${SEED}.csv"

echo "" | tee -a $GRID_LOG
echo "=== Phase 1: TF-IDF (SASRecAlignV3, base) ===" | tee -a $GRID_LOG
for LR in "${LRS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
        run_experiment "SASRecAlignV3" "$ROOT/sasrec_align_base_stratified_v3_ts.yaml" $LR $DO
    done
done

echo "" | tee -a $GRID_LOG
echo "=== Phase 2: LLM (SASRecAlignV3, qwen) ===" | tee -a $GRID_LOG
for LR in "${LRS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
        run_experiment "SASRecAlignV3" "$ROOT/sasrec_align_qwen3_stratified_v3_ts.yaml" $LR $DO
    done
done

echo "" | tee -a $GRID_LOG
echo "=== Phase 3: MV (SASRecAlignMultiViewV3) ===" | tee -a $GRID_LOG
for LR in "${LRS[@]}"; do
    for DO in "${DROPOUTS[@]}"; do
        run_experiment "SASRecAlignMultiViewV3" "$ROOT/sasrec_align_multi_view_v3_stratified_ts.yaml" $LR $DO
    done
done

echo "" | tee -a $GRID_LOG
echo "=== Grid Search completed at $(date) ===" | tee -a $GRID_LOG
echo "" | tee -a $GRID_LOG
echo "Results CSV: logs/grid_results_beauty_seed${SEED}.csv" | tee -a $GRID_LOG

# Print summary
echo "" | tee -a $GRID_LOG
echo "=== SUMMARY (sorted by valid_mrr) ===" | tee -a $GRID_LOG
sort -t',' -k5 -rn "logs/grid_results_beauty_seed${SEED}.csv" | head -10 | tee -a $GRID_LOG
