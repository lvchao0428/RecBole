#!/bin/bash

# 快速测试脚本 - 验证alignment_weight调整的效果
# 不做grid search，直接用推荐配置跑一次

echo "======================================"
echo "快速测试：白化特征 + 降低text依赖"
echo "======================================"
echo ""
echo "配置改动："
echo "  - alignment_weight: 0.1 → 0.05"
echo "  - cross_dropout_prob: 0.0 → 0.3"
echo "  - token_dropout_prob: 0.1 → 0.15"
echo "  - text_weight: 0.4 → 0.3"
echo ""
echo "预期效果："
echo "  - Recall@10: 0.0511 → ≥0.065 (恢复召回能力)"
echo "  - MRR@10: 保持 ≥0.027 (保留排序优势)"
echo ""

python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files sasrec_align_base_whiten_v2.yaml

echo ""
echo "======================================"
echo "测试完成，请查看结果："
echo "  - 如果Recall恢复，说明确实是text依赖过高的问题"
echo "  - 如果仍然Recall低，可能需要进一步降低alignment_weight到0.03"
echo "======================================"

