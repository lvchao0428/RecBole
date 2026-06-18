#!/usr/bin/env bash
# Shared env for local + 5090 training scripts.
# Usage: source "$ROOT/scripts/recbole_env.sh"

if [[ -n "${RECBOLE_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export RECBOLE_ENV_LOADED=1

if [[ -x "${HOME}/anaconda3/bin/python" ]]; then
  export PATH="${HOME}/anaconda3/bin:${PATH}"
elif ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  export PATH="$(dirname "$(command -v python3)"):${PATH}"
fi
