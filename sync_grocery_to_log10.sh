#!/usr/bin/env bash
# 5090 → log10：同步 Grocery ID-only 所需代码与数据（不含 embedding）

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

if [[ -z "${LOG10_SSH}" ]]; then
  echo "ERROR: LOG10_SSH not set. Edit scripts/log10_env.sh or export LOG10_SSH=user@host"
  exit 1
fi

DS="Amazon_Grocery_and_Gourmet_Food"
REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"

echo ">>> rsync Grocery ID-only bundle → ${REMOTE}"

ssh "$LOG10_SSH" "mkdir -p ${LOG10_PROJECT}/dataset/${DS} ${LOG10_PROJECT}/logs ${LOG10_PROJECT}/saved ${LOG10_PROJECT}/run_metrics ${LOG10_PROJECT}/recbole ${LOG10_PROJECT}/scripts"

rsync -avz --progress \
  "$ROOT/run_recbole.py" \
  "$ROOT/run_log10_grocery_id_only.sh" \
  "$ROOT/run50epBase_grocery_stratified.sh" \
  "$ROOT/sasrec_baseline_50ep_grocery_stratified.yaml" \
  "$ROOT/sasrec_grocery_plain.yaml" \
  "${REMOTE}/"

rsync -avz --progress "$ROOT/recbole/" "${REMOTE}/recbole/"
rsync -avz --progress "$ROOT/scripts/two_phase_train.py" "$ROOT/scripts/recbole_env.sh" "${REMOTE}/scripts/"

rsync -avz --progress \
  "$ROOT/dataset/${DS}/${DS}.inter" \
  "$ROOT/dataset/${DS}/${DS}.item" \
  "${REMOTE}/dataset/${DS}/"

ssh "$LOG10_SSH" "chmod +x ${LOG10_PROJECT}/run_log10_grocery_id_only.sh"

echo "✅ sync done"
