#!/usr/bin/env bash
# 汇总 BC + Grocery 实验 metrics → paper_recsys/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/paper_recsys/experiment_summary_$(date +%Y%m%d).md"

{
  echo "# 实验汇总 $(date +%Y-%m-%d)"
  echo
  echo "## book-crossing (5090, seed=2024 phaseb50 + 历史 seed2025)"
  echo '```'
  python3 tools/pull_bc_metrics.py 2>/dev/null || echo "(no BC metrics)"
  echo '```'
  echo
  echo "## Amazon Grocery (5090 text + log10 ID)"
  echo '```'
  python3 tools/pull_grocery_metrics.py 2>/dev/null || echo "(no Grocery metrics yet)"
  echo '```'
} > "$OUT"

echo "Wrote $OUT"
cat "$OUT"
