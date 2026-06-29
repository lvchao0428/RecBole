#!/usr/bin/env bash
# Sync entire RecBole project to the 5090 training server.
#
# Usage:
#   ./sync_recbole.sh              # 正常同步
#   ./sync_recbole.sh --dry-run    # 预览，不实际传输
#   REMOTE_HOST=xx.xx.xx.xx ./sync_recbole.sh   # 指定服务器

set -euo pipefail

# 同步源：项目根目录（脚本所在位置）
ROOT="${SYNC_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

REMOTE_USER="${REMOTE_USER:-charlie}"
REMOTE_HOST="${REMOTE_HOST:-www.ultrapp.online}"
REMOTE_DIR="${REMOTE_DIR:-/home/charlie/project/RecBole}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    *)
      echo "Unknown option: $arg (only --dry-run / -n supported)" >&2
      exit 1
      ;;
  esac
done

RSYNC=(rsync -avz --human-readable)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC+=(--dry-run)
fi

RSYNC+=(
  --delete
  # 完全不触碰服务器端数据目录（避免 --delete 镜像时误删 .inter/.item 等）
  --exclude "dataset/"
  --exclude "release/dataset/"
  # 保护训练产物
  --filter "P saved/"
  --filter "P log/"
  --filter "P run_metrics/"
  --filter "P results/"
  --exclude "run_metrics/**"
  --exclude "log_tensorboard/**"
  # Python 缓存
  --exclude "__pycache__/"
  --exclude "*.py[cod]"
  --exclude "*.egg-info/"
  --exclude ".venv/"
  # macOS
  --exclude ".DS_Store"
  # 大文件 / 产物（服务器端自行生成，不从本地推送）
  --exclude "*.zip"
  --exclude "*.pkl"
  --exclude "*.npy"
  --exclude "*.npz"
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.safetensors"
  --exclude "log/"
  --exclude "logs/"
  --exclude "saved/"
  # 论文 PDF（太大，不需要同步到训练服务器）
  --exclude "paper_recsys/*.pdf"
)

echo "Sync: $ROOT/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
"${RSYNC[@]}" "$ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Done."
