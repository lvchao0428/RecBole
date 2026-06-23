#!/usr/bin/env bash
# log10 (1080Ti 11GB) — 仅 5090 内网可达，本机无法 ssh
#
# 5090 交互 shell: alias log10="ssh charlie@192.168.0.137"
# 非交互脚本/nohup 不能依赖 alias → 直接用 IP

export LOG10_SSH="${LOG10_SSH:-charlie@192.168.0.137}"
export LOG10_PROJECT="${LOG10_PROJECT:-~/project/RecBole}"
export LOG10_PYTHON="${LOG10_PYTHON:-python3}"
