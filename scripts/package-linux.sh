#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist/krita-smart-select}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
PACKAGE_NAME="krita-smart-select-linux.tar.gz"

if [[ ! -d "$DIST_DIR" ]]; then
  echo "Distribution directory does not exist: $DIST_DIR" >&2
  echo "Run scripts/build-krita-5.3.1.sh first." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
PACKAGE_ROOT="$(mktemp -d -t krita-smart-select-package-XXXXXX)"
trap 'rm -rf "$PACKAGE_ROOT"' EXIT

rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$DIST_DIR/" \
  "$PACKAGE_ROOT/$(basename "$DIST_DIR")/"

cp "$ROOT_DIR/README.md" "$PACKAGE_ROOT/$(basename "$DIST_DIR")/"
cp "$ROOT_DIR/CONTRIBUTING.md" "$PACKAGE_ROOT/$(basename "$DIST_DIR")/"
cp "$ROOT_DIR/LICENSE" "$PACKAGE_ROOT/$(basename "$DIST_DIR")/"
rsync -a "$ROOT_DIR/docs/" "$PACKAGE_ROOT/$(basename "$DIST_DIR")/docs/"

tar \
  -C "$PACKAGE_ROOT" \
  -czf "$OUT_DIR/$PACKAGE_NAME" \
  "$(basename "$DIST_DIR")"
echo "$OUT_DIR/$PACKAGE_NAME"
