#!/bin/bash

# 白化特征两阶段训练 - 修正版
# 
# 核心问题诊断：
# 1. 白化改变了特征几何结构（主成分被拉平）
# 2. 模型过度自信于text signal → MRR↑ 但 Recall↓
# 3. Phase-A可能过拟合（删除early stop后跑满8 epochs）
#
# 修正策略：
# 1. 缩短Phase-A到5 epochs，避免过拟合
# 2. 降低alignment_weight (0.03/0.05/0.08)，减少text依赖
# 3. 使用NDCG@10作为gate（比MRR更平衡）
# 4. Temperature grid包含原值0.07
# 5. 增加cross_dropout防止fusion模块过拟合

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_whiten_v2.yaml" \
  \
  --phase_a_epochs 5 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "NDCG@10" \
  \
  --phase_b_epochs 40 \
  \
  --lr_text_head 0.001 \
  --lr_dnn_cross 0.0005 \
  --backbone_lr_scale 0.1 \
  \
  --phase_a_grid \
  --align_grid "0.03,0.05,0.08" \
  --tau_grid "0.05,0.07,0.1" \
  \
  --ndcg_baseline 0.0342 \
  --ndcg_gain_threshold 0.015 \
  --phase_a_auto_to_b \
  \
  --save \
  --checkpoint_dir "./saved/whiten_revised"

# 预期改进：
# - Recall@10: 从 0.0511 恢复到 ≥0.065
# - MRR@10: 保持在 ≥0.027 (保留白化的好处)
# - NDCG@10: 提升到 ≥0.036 (综合提升)

