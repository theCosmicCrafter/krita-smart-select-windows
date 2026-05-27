#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist/krita-smart-select}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
PACKAGE_NAME="krita-smart-select-linux-mvp.tar.gz"

if [[ ! -d "$DIST_DIR" ]]; then
  echo "Distribution directory does not exist: $DIST_DIR" >&2
  echo "Run scripts/build-krita-5.3.1.sh first." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
tar -C "$DIST_DIR/.." -czf "$OUT_DIR/$PACKAGE_NAME" "$(basename "$DIST_DIR")"
echo "$OUT_DIR/$PACKAGE_NAME"

