#!/usr/bin/env bash
#set -euo pipefail

# Resume Phase-B training from checkpoint (TF-IDF+LLM, Movies)
# 从 checkpoint 继续训练 Phase-B

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ========================================
# 重要：设置你的 checkpoint 路径
# ========================================
# 请将下面的路径替换为你实际的 checkpoint 文件路径
# 例如：CHECKPOINT_PATH="./saved/phase_runs_movies_stratified/SASRec_Align-Dec-13-2025_10-30-45.pth"
CHECKPOINT_PATH="./saved/phase_runs_movies_stratified/SASRec_Align-Dec-14-2025_10-03-52.pth"

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script to point to your actual checkpoint file."
    exit 1
fi

echo "========================================="
echo "Resume Phase-B: TF-IDF+LLM (Movies)"
echo "========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: Amazon_Movies_and_TV"
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Movies_and_TV \
  --config_files "sasrec_align_movies_qwen3_stratified.yaml" \
  --only_phase_b \
  --resume_from "$CHECKPOINT_PATH" \
  --phase_b_epochs 20 \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_movies_stratified \
  --seed 2025 \
  --variant_features "sasrec,tfidf,llm,movies,stratified,resumed" \
  --watchdog_disable \
  --save

echo ""
echo "✅ Phase-B Resume Training Done!"
echo ""
