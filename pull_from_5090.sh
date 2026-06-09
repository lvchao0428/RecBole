#!/usr/bin/env bash
# Pull release/ (code + configs) from the 5090 training server to local.
#
# Usage:
#   ./pull_from_5090.sh              # 拉取代码和配置
#   ./pull_from_5090.sh --dry-run    # 预览，不实际传输
#   ./pull_from_5090.sh --all        # 连同数据文件一起拉 (npy/pkl/pth 等)
#
# 默认不拉大文件 (npy/pkl/pth/safetensors/zip)，加 --all 可拉全部。

set -euo pipefail

ROOT="${SYNC_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

REMOTE_USER="${REMOTE_USER:-charlie}"
REMOTE_HOST="${REMOTE_HOST:-www.ultrapp.online}"
REMOTE_DIR="${REMOTE_DIR:-/home/charlie/project/RecBole}"

DRY_RUN=0
PULL_ALL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --all|-a)     PULL_ALL=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--dry-run|-n] [--all|-a]" >&2
      exit 1
      ;;
  esac
done

RSYNC=(rsync -avz --human-readable)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC+=(--dry-run)
fi

RSYNC+=(
  --exclude "__pycache__/"
  --exclude "*.py[cod]"
  --exclude "*.egg-info/"
  --exclude ".venv/"
  --exclude ".DS_Store"
  --exclude "log/"
  --exclude "saved/"
)

if [[ "$PULL_ALL" -eq 0 ]]; then
  RSYNC+=(
    --exclude "*.zip"
    --exclude "*.pkl"
    --exclude "*.npy"
    --exclude "*.npz"
    --exclude "*.pt"
    --exclude "*.pth"
    --exclude "*.safetensors"
  )
  echo "Mode: code + configs only (skip large data files)"
  echo "  Use --all to include npy/pkl/pth/safetensors"
else
  echo "Mode: full pull (including data files)"
fi

echo ""
echo "Pull: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/release/ -> ${ROOT}/release/"
"${RSYNC[@]}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/release/" "${ROOT}/release/"
echo ""
echo "Done."
