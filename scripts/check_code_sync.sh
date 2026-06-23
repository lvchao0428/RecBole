#!/usr/bin/env bash
# 检查 5090 / log10 / 本地 关键文件 MD5（以当前目录=5090 或本地为准）
# Usage:
#   bash scripts/check_code_sync.sh           # 在 5090 或本地跑
#   CHECK_LOG10=1 bash scripts/check_code_sync.sh  # 5090 上同时检查 log10

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FILES=(
  run_recbole.py
  scripts/two_phase_train.py
  scripts/recbole_env.sh
  scripts/log10_env.sh
  scripts/sync_grocery_to_log10.sh
  scripts/sync_log10_code_from_5090.sh
  run50epBase_book_crossing_stratified.sh
  run50epBase_grocery_stratified.sh
  run_log10_grocery_id_only.sh
  run_5090_grocery_pipeline.sh
  run_5090_queue_after_bc_grocery.sh
  run_5090_queue_bc_four_configs_phaseb50.sh
  sasrec_baseline_50ep_book_crossing_stratified.yaml
  sasrec_baseline_50ep_grocery_stratified.yaml
  sasrec_grocery_plain.yaml
  two_phase_run_tfidf_v3_grocery_stratified.sh
  two_phase_run_tfidf_llm_v3_grocery_stratified.sh
  two_phase_run_multiview_v3_grocery_stratified_7b.sh
  tools/setup_grocery_dataset.sh
  tools/gen_text_emb_grocery_qwen2.5_7b.sh
  recbole/quick_start/quick_start.py
  recbole/trainer/trainer.py
)

md5f() {
  md5sum "$1" 2>/dev/null | awk '{print $1}' || md5 -q "$1" 2>/dev/null || echo MISSING
}

echo "HOST=$(hostname) GIT=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
printf "%-55s %s\n" "FILE" "MD5"
for f in "${FILES[@]}"; do
  printf "%-55s %s\n" "$f" "$(md5f "$f")"
done

if [[ "${CHECK_LOG10:-0}" == "1" && -f "$ROOT/scripts/log10_env.sh" ]]; then
  source "$ROOT/scripts/log10_env.sh"
  echo ""
  echo "=== log10 ($LOG10_SSH) GIT=$(ssh "$LOG10_SSH" "cd ${LOG10_PROJECT} && git rev-parse --short HEAD 2>/dev/null || echo none") ==="
  for f in "${FILES[@]}"; do
    remote=$(ssh "$LOG10_SSH" "md5sum ${LOG10_PROJECT}/$f 2>/dev/null | awk '{print \$1}' || echo MISSING")
    local=$(md5f "$f")
    flag=""
    [[ "$remote" != "$local" ]] && flag="  <-- DIFF"
    printf "%-55s %s%s\n" "$f" "$remote" "$flag"
  done
fi
