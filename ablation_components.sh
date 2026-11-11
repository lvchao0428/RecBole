#!/bin/bash
# 消融实验：分离各组件的贡献

cd "$(dirname "$0")/.." || exit 1

echo "=== 消融实验：分离各组件贡献 ==="
echo ""

# 基础配置
BASE_CONFIG="sasrec_base_plain.yaml"
DATASET="Amazon_Beauty"
EPOCHS=30

run_experiment() {
    local name=$1
    local config_dict=$2
    local desc=$3
    
    echo "运行: $name - $desc"
    python run_recbole.py \
        --model SASRec_Align \
        --dataset $DATASET \
        --config_files $BASE_CONFIG \
        --config_dict "$config_dict;epochs=$EPOCHS;eval_step=5" \
        > results/${name}.log 2>&1
}

# 消融实验矩阵
echo "实验设计："
echo "1. baseline: 纯ID"
echo "2. +text: 添加文本特征（简单融合）"
echo "3. +cross: 使用交叉网络融合"
echo "4. +align: 添加对齐损失"
echo "5. +reg: 添加正则化"
echo ""

# 运行实验
run_experiment "ablation_1_baseline" \
    "disable_text_feature=true" \
    "纯ID基线"

run_experiment "ablation_2_text" \
    "use_cross=false;use_align=false" \
    "简单文本融合"

run_experiment "ablation_3_cross" \
    "use_cross=true;use_align=false" \
    "交叉网络融合"

run_experiment "ablation_4_align" \
    "use_cross=true;use_align=true;alignment_weight=0.1" \
    "交叉+对齐"

run_experiment "ablation_5_reg" \
    "use_cross=true;use_align=true;alignment_weight=0.1;text_gate_reg_l2=0.005;cross_dropout_prob=0.1" \
    "全部特性+正则"

# 汇总结果
echo -e "\n=== 消融实验结果汇总 ==="
for i in {1..5}; do
    log="results/ablation_${i}_*.log"
    if [ -f $log ]; then
        echo -e "\n实验$i:"
        grep "test result" $log | tail -1 | grep -oE "MRR@10.*|Recall@10.*"
    fi
done
