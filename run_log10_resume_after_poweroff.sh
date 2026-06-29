#!/usr/bin/env bash
# log10 断电/手动停止后恢复 GRU4Rec baseline。
# 从 seed=2024 整块重跑（Beauty+Toys ID+TF-IDF）；seed=42 已完成。
#
# Usage:
#   nohup bash run_log10_resume_after_poweroff.sh >> logs/log10_resume_after_poweroff_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash "$ROOT/run_log10_gru4rec_baselines_from_2024.sh"
