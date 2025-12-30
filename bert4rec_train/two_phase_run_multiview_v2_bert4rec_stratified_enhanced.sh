#!/usr/bin/env bash

# Enhanced Multi-View V2 Training for BERT4Rec
# 针对BERT4Rec优化的Multi-View配置
#
# 主要改进:
# ============================================================
# 1. 延长预热期 (backbone_burnin_epochs: 10 → 15)
#    - BERT4Rec需要更长时间建立有效的ID表示
#
# 2. 提高Phase A训练轮数 (15 → 20)
#    - 让Text特征有更多时间与ID表示对齐
#
# 3. 提高Phase B训练轮数 (40 → 55)
#    - 充分利用Multi-View特征
#
# 4. 降低backbone_lr_scale (0.1 → 0.05)
#    - 防止破坏已学到的ID表示
#
# 5. 提高Text头学习率 (1e-3 → 2e-3)
#    - 加速Text特征融合学习
#
# 6. 增强对齐权重 (0.05 → 0.1)
#    - 让Multi-View对齐信号更强
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "BERT4Rec Multi-View V2 ENHANCED"
echo "========================================="
echo "Key Enhancements for BERT4Rec:"
echo "  - Longer warmup (15 epochs)"
echo "  - Extended Phase A (20 epochs)"  
echo "  - Extended Phase B (55 epochs)"
echo "  - Lower backbone LR scale (0.05)"
echo "  - Higher text head LR (2e-3)"
echo "  - Stronger alignment (0.1)"
echo "  - multiview_align_scale: 3.0"
echo "  - text_view_senet_ratio: 1 (no compression)"
echo "  - text_weight: 1.2 (amplified)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

base_dir='/home/charlie/project/RecBole'
#base_dir='/home/ubuntu/own/RecBole/'
cd ${base_dir}
#yaml_dir='/home/ubuntu/own/RecBole/bert4rec_train'
yaml_dir='/home/charlie/project/RecBole/bert4rec_train'

python scripts/two_phase_train.py \
  --model BERT4RecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "${yaml_dir}/bert4rec_align_multi_view_v2_stratified_enhanced.yaml" \
  --phase_a_grid \
  --align_grid "0.05" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 15 \
  --burnin_eval_step 3 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 2 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0038 \
  --metric_gain_threshold 0.005 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 1e-3 \
  --phase_a_text_gate_reg_l2 0.03 \
  --phase_b_alignment_weight 0.1 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.2 \
  --phase_a_auto_to_b \
  --phase_b_epochs 55 \
  --backbone_lr_scale 0.05 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_bert4rec_enhanced \
  --seed 2025 \
  --variant_features "bert4rec,multiview_v2_enhanced,4views,senet_ratio_1,align_scale_3x,text_weight_1.2" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "Enhanced Settings Summary:"
echo "  - n_layers: 3 (vs 2 in baseline)"
echo "  - mask_ratio: 0.15 (vs 0.2)"
echo "  - multiview_align_scale: 3.0 (vs 2.0)"
echo "  - text_view_senet_ratio: 1 (vs 2, no compression)"
echo "  - text_weight: 1.2 (vs 0.8)"
echo "  - cold_start_align_boost: 5.0 (vs 3.0)"
echo "  - dropout: 0.3 (vs 0.5)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10"
echo ""

