#!/usr/bin/env bash
# Pure SASRec baseline (60 epochs) with Stratified Evaluation (book-crossing)
# 与 two_phase_*_book_crossing 中 Phase A 20 + Phase B 40 对齐（合计 60）
#
# 在项目根目录执行。随机种子由环境变量 SEED 传入；未设置时默认 2025。
# 多 seed 批量：见 run60epBase_book_crossing_stratified_multiseed.sh 或
# run_all_book_crossing_stratified_sequential_multiseed.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${SEED:-2025}"

echo "========================================="
echo "SASRec Baseline (60ep) with Stratified Metrics (book-crossing)"
echo "========================================="
echo "Dataset: book-crossing"
echo "Model: Pure SASRec (no text features)"
echo "SEED: $SEED"
echo "checkpoint_dir: saved/baseline_60ep_book_crossing_stratified_seed${SEED}"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

export SEED
python <<'PY'
import os
from recbole.quick_start import run

seed = int(os.environ["SEED"])
run(
    "SASRecAlign",
    "book-crossing",
    config_file_list=["sasrec_baseline_60ep_book_crossing_stratified.yaml"],
    config_dict={
        "seed": seed,
        "checkpoint_dir": f"saved/baseline_60ep_book_crossing_stratified_seed{seed}",
    },
)
PY

echo ""
echo "✅ Training Done (seed=$SEED). Check results for stratified metrics."
echo ""
