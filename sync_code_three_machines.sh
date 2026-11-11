#!/usr/bin/env bash
# Sync code from local → 5090 → log10 (local is source of truth)
# Run from local Mac:
#   bash scripts/sync_code_three_machines.sh
#
# Excludes: dataset/, logs/, saved/, run_metrics/, .git/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE_5090="${REMOTE_5090:-charlie@www.ultrapp.online}"
REMOTE_5090_DIR="${REMOTE_5090_DIR:-/home/charlie/project/RecBole}"
LOG10_SSH="${LOG10_SSH:-charlie@192.168.0.107}"
LOG10_PROJECT="${LOG10_PROJECT:-/home/charlie/project/RecBole}"

RSYNC_EXCLUDE=(
  --exclude '.git/'
  --exclude 'dataset/'
  --exclude 'logs/'
  --exclude 'saved/'
  --exclude 'run_metrics/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

echo ">>> [1/2] Local → 5090"
rsync -avz "${RSYNC_EXCLUDE[@]}" \
  "$ROOT/recbole/" "${REMOTE_5090}:${REMOTE_5090_DIR}/recbole/"
rsync -avz "${RSYNC_EXCLUDE[@]}" \
  "$ROOT/scripts/" "${REMOTE_5090}:${REMOTE_5090_DIR}/scripts/"
rsync -avz "${RSYNC_EXCLUDE[@]}" \
  "$ROOT/tools/" "${REMOTE_5090}:${REMOTE_5090_DIR}/tools/"
rsync -avz \
  "$ROOT/run_"*.sh \
  "$ROOT/sasrec_"*.yaml \
  "$ROOT/unisrec_"*.yaml \
  "$ROOT/new_watcher_0713.sh" \
  "${REMOTE_5090}:${REMOTE_5090_DIR}/" 2>/dev/null || true
rsync -avz "${RSYNC_EXCLUDE[@]}" \
  "$ROOT/paper_recsys/" "${REMOTE_5090}:${REMOTE_5090_DIR}/paper_recsys/"
echo "✅ 5090 sync done"

echo ">>> [2/2] 5090 → log10 (via 5090 relay)"
ssh "$REMOTE_5090" bash -s <<EOF
set -euo pipefail
ROOT="$REMOTE_5090_DIR"
LOG10="$LOG10_SSH:$LOG10_PROJECT"
RSYNC_EXCLUDE=(--exclude '.git/' --exclude 'dataset/' --exclude 'logs/' --exclude 'saved/' --exclude 'run_metrics/' --exclude '__pycache__/' --exclude '*.pyc')
ssh "$LOG10_SSH" "mkdir -p $LOG10_PROJECT/{scripts,recbole,tools,paper_recsys}"
rsync -avz "\${RSYNC_EXCLUDE[@]}" "\$ROOT/recbole/" "\$LOG10/recbole/"
rsync -avz "\${RSYNC_EXCLUDE[@]}" "\$ROOT/scripts/" "\$LOG10/scripts/"
rsync -avz "\${RSYNC_EXCLUDE[@]}" "\$ROOT/tools/" "\$LOG10/tools/"
rsync -avz "\$ROOT/run_"*.sh "\$ROOT/sasrec_"*.yaml "\$ROOT/unisrec_"*.yaml "\$ROOT/new_watcher_0713.sh" "\$LOG10/" 2>/dev/null || true
rsync -avz "\${RSYNC_EXCLUDE[@]}" "\$ROOT/paper_recsys/" "\$LOG10/paper_recsys/"
echo "✅ log10 sync done"
EOF

echo "======== Three-machine code sync complete ========"
