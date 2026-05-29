#!/usr/bin/env bash
set -euo pipefail

debug_root="${1:-/tmp}"

if [[ ! -d "$debug_root" ]]; then
  echo "Debug root does not exist: $debug_root" >&2
  exit 1
fi

find "$debug_root" -maxdepth 1 -type d -name 'krita-smart-select-debug-*' -print -exec rm -rf {} +
