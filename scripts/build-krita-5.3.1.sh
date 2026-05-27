#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KRITA_SRC="${KRITA_SRC:-/home/bogdan/Work/sources/krita}"
KRITA_BUILD="${KRITA_BUILD:-/home/bogdan/Work/build/krita-5.3.1}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$ROOT_DIR/dist/krita-smart-select}"
BUILD_JOBS="${BUILD_JOBS:-2}"
PLUGIN_LINK="$KRITA_SRC/plugins/tools/krita-smart-select"
TOOLS_CMAKE="$KRITA_SRC/plugins/tools/CMakeLists.txt"

if [[ ! -f "$KRITA_SRC/CMakeLists.txt" ]]; then
  cat >&2 <<EOF
KRITA_SRC does not look like a Krita source tree: $KRITA_SRC

Example:
  KRITA_SRC=/home/bogdan/Work/sources/krita \
  KRITA_BUILD=/home/bogdan/Work/build/krita-5.3.1 \
  ./scripts/build-krita-5.3.1.sh
EOF
  exit 2
fi

ln -sfn "$ROOT_DIR/plugin" "$PLUGIN_LINK"

if ! grep -q 'add_subdirectory( krita-smart-select )' "$TOOLS_CMAKE"; then
  python3 - "$TOOLS_CMAKE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "add_subdirectory( tool_knife )\n"
replacement = needle + "add_subdirectory( krita-smart-select )\n"
if needle not in text:
    raise SystemExit("Could not find insertion point in plugins/tools/CMakeLists.txt")
path.write_text(text.replace(needle, replacement, 1))
PY
fi

cmake -S "$KRITA_SRC" -B "$KRITA_BUILD" \
  -DBUILD_WITH_QT6=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

cmake --build "$KRITA_BUILD" --target kritasmartselect --parallel "$BUILD_JOBS"

mkdir -p "$INSTALL_PREFIX/pykrita/smart_select/lib"
mkdir -p "$INSTALL_PREFIX/share/krita/actions"
cp "$KRITA_BUILD/bin/kritasmartselect.so" "$INSTALL_PREFIX/pykrita/smart_select/lib/"
cp "$ROOT_DIR/python/smart_select.desktop" "$INSTALL_PREFIX/pykrita/"
cp "$ROOT_DIR/plugin/SmartSelectTool.action" "$INSTALL_PREFIX/share/krita/actions/"
rsync -a --include='*/' --include='*.py' --exclude='*' "$ROOT_DIR/python/smart_select/" "$INSTALL_PREFIX/pykrita/smart_select/"
rsync -a --include='*/' --include='*.py' --exclude='*' "$ROOT_DIR/worker/" "$INSTALL_PREFIX/pykrita/smart_select/worker/"

echo "Built Smart Select into: $INSTALL_PREFIX"
