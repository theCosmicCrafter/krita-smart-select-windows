#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KRITA_SRC="${KRITA_SRC:-/home/bogdan/Work/sources/krita}"
KRITA_BUILD="${KRITA_BUILD:-/home/bogdan/Work/build/krita-5.3.1}"
BUILD_JOBS="${BUILD_JOBS:-2}"
PYTHON="${PYTHON:-python3}"
PLUGIN_LINK="$KRITA_SRC/plugins/tools/krita-smart-select"
TOOLS_CMAKE="$KRITA_SRC/plugins/tools/CMakeLists.txt"

cd "$ROOT_DIR"

"$PYTHON" -m py_compile \
  worker/model_cache.py \
  worker/model_registry.py \
  worker/image_preprocessing.py \
  worker/job_runner.py \
  worker/runtime_env.py \
  worker/runtime_diagnostics.py \
  worker/runtime_manifest.py \
  worker/sam2_inference.py \
  worker/segment_map_builder.py \
  worker/smart_select_worker.py \
  python/smart_select/extension.py \
  python/smart_select/worker_launcher.py \
  python/smart_select/__init__.py \
  scripts/render-segment-map.py

if ! "$PYTHON" -m mypy --version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
mypy is not installed.

Install development dependencies first, for example:
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements-dev.txt
  PYTHON=.venv/bin/python ./scripts/test-code.sh

Or run this script from an environment where `$PYTHON -m mypy` is available.
EOF
  exit 2
fi

if ! "$PYTHON" -m ruff --version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ruff is not installed.

Install development dependencies first, for example:
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements-dev.txt
  PYTHON=.venv/bin/python ./scripts/test-code.sh

Or run this script from an environment where `$PYTHON -m ruff` is available.
EOF
  exit 2
fi

"$PYTHON" -m ruff format --check python worker tests
"$PYTHON" -m ruff check python worker tests
"$PYTHON" -m mypy

"$PYTHON" -m unittest discover -s tests/worker -p 'test_*.py'

if [[ ! -f "$KRITA_SRC/CMakeLists.txt" ]]; then
  echo "Skipping C++ tests: KRITA_SRC does not look like a Krita source tree: $KRITA_SRC" >&2
  exit 0
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
  -DSMART_SELECT_BUILD_TESTS=ON

cmake --build "$KRITA_BUILD" --target smartselect_core_tests --parallel "$BUILD_JOBS"
ctest --test-dir "$KRITA_BUILD" -R '^smartselect_core_tests$' --output-on-failure
