#!/usr/bin/env bash
set -euo pipefail

SAM3_ENV_NAME="${SAM3_ENV_NAME:-sam3-mps}"
SAM3_SOURCE="${SAM3_SOURCE:-$HOME/.cache/dream2detect/sam3-apple-silicon-support-v2}"

if [[ ! -d "$SAM3_SOURCE/sam3" ]]; then
  echo "SAM3 source checkout not found at: $SAM3_SOURCE" >&2
  echo "Clone the Apple Silicon branch or set SAM3_SOURCE to its checkout path." >&2
  exit 1
fi

export PYTHONPATH="$SAM3_SOURCE:${PYTHONPATH:-}"

conda run -n "$SAM3_ENV_NAME" python "$@"
