#!/bin/bash
# 公平对比实验：确保InfoNCE不影响纯ID模型

cd "$(dirname "$0")/.." || exit 1

echo "=== 公平对比实验 ==="
echo "验证InfoNCE只在有文本特征时生效"
echo ""

# 实验1: 纯ID + align=true（验证InfoNCE不会执行）
echo "1. 纯ID模型 + align=true（应该无影响）"
python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files sasrec_base_plain.yaml \
  --config_dict "disable_text_feature=true;use_align=true;alignment_weight=0.1;epochs=10" \
  > results/pure_id_with_align.log 2>&1 &

# 实验2: 纯ID + align=false（基线）
echo "2. 纯ID模型 + align=false（基线）"
python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files sasrec_base_plain.yaml \
  --config_dict "disable_text_feature=true;use_align=false;epochs=10" \
  > results/pure_id_no_align.log 2>&1 &

wait

# 比较结果
echo -e "\n=== 结果对比 ==="
echo "如果InfoNCE正确实现，两个实验的结果应该完全相同："
echo ""

echo "实验1（align=true）："
grep -A 5 "test result" results/pure_id_with_align.log | tail -6

echo -e "\n实验2（align=false）："
grep -A 5 "test result" results/pure_id_no_align.log | tail -6

# 检查是否有对齐损失的日志
echo -e "\n=== 检查对齐损失 ==="
echo "实验1中的align相关日志（应该为空）："
grep -i "align" results/pure_id_with_align.log | grep -v "config" | head -5
