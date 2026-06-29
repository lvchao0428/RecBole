#!/usr/bin/env bash
# 5090 断电/手动停止后恢复 Phase 4 + Phase 5。
# 从 Toys seed=2025 整组重跑（LLM+MV）；Beauty 42/2024/2025 已完成，不重复。
#
# Usage:
#   nohup bash run_5090_resume_after_poweroff.sh >> logs/resume_after_poweroff_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash "$ROOT/run_5090_phase4_text_resume_from_toys2025.sh"
