#!/bin/bash
# 融合机制对比实验

cd "$(dirname "$0")/.." || exit 1

echo "=== 融合机制对比实验 ==="
echo ""

# 实验1: 简单拼接（当前方式）
echo "1. 简单拼接融合 (concat)"
python scripts/simple_check_text.py \
  sasrec_llm_plain.yaml \
  > results/fusion_concat.log 2>&1
grep -E "(有效权重|融合差异)" results/fusion_concat.log

# 实验2: 交叉网络融合
echo -e "\n2. DCN-V2交叉融合"
python scripts/simple_check_text.py \
  sasrec_llm_plain.yaml \
  --config_dict "use_cross=true" \
  > results/fusion_cross.log 2>&1
grep -E "(有效权重|融合差异)" results/fusion_cross.log

# 实验3: 交叉+对齐
echo -e "\n3. 交叉+对齐融合"
python scripts/simple_check_text.py \
  sasrec_llm_plain.yaml \
  --config_dict "use_cross=true;use_align=true;alignment_weight=0.1" \
  > results/fusion_cross_align.log 2>&1
grep -E "(有效权重|融合差异)" results/fusion_cross_align.log

echo -e "\n=== 总结 ==="
echo "如果融合差异都很大但性能相似，说明需要："
echo "1. 使用更强的融合机制（cross）"
echo "2. 添加对齐损失（align）"
echo "3. 调整文本权重和正则化"
