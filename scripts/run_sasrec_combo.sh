#!/bin/bash
# 通用启动脚本：按给定 YAML 配置运行一场 SASRec_Align 实验。
# 用法：
#   bash scripts/run_sasrec_combo.sh  <yaml_cfg>  [dataset]
#   示例：
#     bash scripts/run_sasrec_combo.sh sasrec_plain.yaml             # 默认 Amazon_Beauty
#     bash scripts/run_sasrec_combo.sh sasrec_llm_cross.yaml  MIND   # 指定数据集
set -e

CFG_FILE=$1
if [[ -z "$CFG_FILE" ]]; then
  echo "[Usage] bash scripts/run_sasrec_combo.sh <yaml_cfg> [dataset]"; exit 1
fi
if [[ ! -f "$CFG_FILE" ]]; then
  echo "[Error] config file '$CFG_FILE' not found!"; exit 1
fi

DATASET=${2:-Amazon_Beauty}
MODEL="SASRec_Align"

# 项目根目录 (脚本所在目录上一级)
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$BASE_DIR"

EXP_NAME="${CFG_FILE%.yaml}"
LOG_DIR="results/sasrec_experiments"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Experiment : $EXP_NAME"
echo "Dataset    : $DATASET"
echo "Config     : $CFG_FILE"
echo "Start time : $(date)"
python run_recbole.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --config_files "$CFG_FILE" \
  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

status=$?
if [[ $status -eq 0 ]]; then
  echo "✓ Finished   $(date)"
  echo "--- Test summary ---"
  grep -A 5 "test result" "$LOG_DIR/${EXP_NAME}.log" | tail -6 || true
else
  echo "✗ Failed (exit code=$status) -- see $LOG_DIR/${EXP_NAME}.log"
fi
echo "Log saved to $LOG_DIR/${EXP_NAME}.log"
