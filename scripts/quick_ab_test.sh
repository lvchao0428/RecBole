#!/bin/bash
# 快速A/B测试：对比纯ID vs 文本特征

echo "=== 快速A/B测试：验证文本特征是否生效 ==="
echo ""

# 测试1：纯ID基线
echo "测试1: 纯ID模型（disable_text_feature=true）"
python scripts/check_text_feature.py \
  --config sasrec_base_plain.yaml overrides/sasrec_plain_pure_idonly.yaml \
  > test1_pure_id.log 2>&1

echo "结果保存到: test1_pure_id.log"
grep -E "(文本模式|有效文本权重|融合嵌入与原始嵌入|总参数量)" test1_pure_id.log

echo ""
echo "----------------------------------------"
echo ""

# 测试2：文本特征（正常权重）
echo "测试2: 文本特征模型（正常权重）"
python scripts/check_text_feature.py \
  --config sasrec_base_plain.yaml overrides/sasrec_plain_pure_base.yaml \
  > test2_text_normal.log 2>&1

echo "结果保存到: test2_text_normal.log"
grep -E "(文本模式|有效文本权重|融合嵌入与原始嵌入|总参数量)" test2_text_normal.log

echo ""
echo "----------------------------------------"
echo ""

# 测试3：文本特征（极端权重）
echo "测试3: 文本特征模型（极端权重）"
python scripts/check_text_feature.py \
  --config sasrec_base_plain.yaml overrides/debug_text_extreme.yaml \
  > test3_text_extreme.log 2>&1

echo "结果保存到: test3_text_extreme.log"
grep -E "(文本模式|有效文本权重|融合嵌入与原始嵌入|总参数量)" test3_text_extreme.log

echo ""
echo "=== 总结 ==="
echo "如果文本特征正常工作，应该看到："
echo "1. test1的'融合嵌入与原始嵌入的平均差异'接近0"
echo "2. test2和test3的差异值明显大于0"
echo "3. test3的差异值应该最大（因为文本权重最大）"
echo "4. 参数数量：test2和test3 > test1"
