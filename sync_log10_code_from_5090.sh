#!/usr/bin/env bash
# 5090 → log10 代码同步（以 5090 为准）
# 在 5090 上执行: bash scripts/sync_log10_code_from_5090.sh
# 不 sync 数据 embedding / saved / logs

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"
DRY="${DRY_RUN:-0}"

echo ">>> sync 5090 → log10 ($LOG10_SSH)"
RSYNC=(rsync -avz --progress)
[[ "$DRY" == "1" ]] && RSYNC+=(--dry-run)

ssh "$LOG10_SSH" "mkdir -p ${LOG10_PROJECT}/{scripts,recbole,logs,saved,run_metrics}"

"${RSYNC[@]}" \
  "$ROOT/run_recbole.py" \
  "$ROOT/run"*.sh \
  "$ROOT/sasrec_"*.yaml \
  "$ROOT/two_phase_run_"*.sh \
  "${REMOTE}/"

"${RSYNC[@]}" \
  "$ROOT/tools/setup_grocery_dataset.sh" \
  "$ROOT/tools/gen_text_emb_grocery_qwen2.5_7b.sh" \
  "$ROOT/tools/pull_bc_metrics.py" \
  "$ROOT/tools/pull_grocery_metrics.py" \
  "$ROOT/tools/summarize_bc_grocery_results.sh" \
  "${REMOTE}/tools/"

"${RSYNC[@]}" "$ROOT/scripts/" "${REMOTE}/scripts/"
"${RSYNC[@]}" "$ROOT/recbole/" "${REMOTE}/recbole/"

ssh "$LOG10_SSH" "chmod +x ${LOG10_PROJECT}/run_log10_grocery_id_only.sh ${LOG10_PROJECT}/scripts/*.sh 2>/dev/null || true"

echo "✅ log10 code sync done"
