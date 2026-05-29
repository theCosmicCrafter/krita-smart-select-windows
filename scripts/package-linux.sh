#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist/krita-smart-select}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
VERSION="${1:-dev}"
ARCHIVE_BASENAME="krita-smart-select-linux-x64-$VERSION"
TARBALL_NAME="$ARCHIVE_BASENAME.tar.gz"
PLUGIN_ZIP_NAME="$ARCHIVE_BASENAME.zip"

if [[ ! -d "$DIST_DIR" ]]; then
  echo "Distribution directory does not exist: $DIST_DIR" >&2
  echo "Run scripts/build-krita-5.3.1.sh first." >&2
  exit 2
fi

if [[ ! -f "$DIST_DIR/pykrita/smart_select.desktop" ]]; then
  echo "Missing Python plugin desktop file in: $DIST_DIR/pykrita" >&2
  echo "Run scripts/build-krita-5.3.1.sh first." >&2
  exit 2
fi

if [[ ! -d "$DIST_DIR/pykrita/smart_select" ]]; then
  echo "Missing Python plugin package in: $DIST_DIR/pykrita/smart_select" >&2
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
  -czf "$OUT_DIR/$TARBALL_NAME" \
  "$(basename "$DIST_DIR")"

PLUGIN_ROOT="$PACKAGE_ROOT/python-plugin"
mkdir -p "$PLUGIN_ROOT"
rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$DIST_DIR/pykrita/smart_select.desktop" \
  "$PLUGIN_ROOT/"
rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$DIST_DIR/pykrita/smart_select" \
  "$PLUGIN_ROOT/"
cp "$DIST_DIR/share/krita/actions/SmartSelectTool.action" \
  "$PLUGIN_ROOT/smart_select/SmartSelectTool.action"

(
  cd "$PLUGIN_ROOT"
  python3 - "$OUT_DIR/$PLUGIN_ZIP_NAME" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

output = Path(sys.argv[1])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(Path(".").rglob("*")):
        if path.is_file():
            archive.write(path, path.as_posix())
PY
)

echo "$OUT_DIR/$PLUGIN_ZIP_NAME"
echo "$OUT_DIR/$TARBALL_NAME"
