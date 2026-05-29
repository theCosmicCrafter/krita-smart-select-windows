#!/usr/bin/env bash
set -euo pipefail

if [[ "${SMART_SELECT_RUN_BIREFNET_SMOKE:-}" != "1" ]]; then
  cat >&2 <<'EOF'
This smoke test may create the managed runtime and download BiRefNet HR Matting.

Run explicitly:
  SMART_SELECT_RUN_BIREFNET_SMOKE=1 ./scripts/test-birefnet-smoke.sh
EOF
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK_DIR="$(mktemp -d -t krita-smart-select-birefnet-smoke-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

export SMART_SELECT_KEEP_DEBUG_FILES=1

"$PYTHON" - "$ROOT_DIR" "$WORK_DIR" <<'PY'
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1])
work = Path(sys.argv[2])


def npy_bytes(descr: str, shape: tuple[int, int], data: bytes) -> bytes:
    header = (
        f"{{'descr': '{descr}', 'fortran_order': False, "
        f"'shape': ({shape[0]}, {shape[1]}), }}"
    )
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data


width, height = 96, 96
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((24, 18, 72, 78), fill=(30, 30, 30))
image_path = work / "snapshot.png"
image.save(image_path)

alpha_path = work / "alpha.npy"
alpha_path.write_bytes(npy_bytes("|u1", (height, width), b"\xff" * (width * height)))

region = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(region)
draw.ellipse((12, 8, 84, 88), fill=255)
region_path = work / "region.npy"
region_path.write_bytes(npy_bytes("|u1", (height, width), region.tobytes()))

request = {
    "id": 1,
    "method": "prepare_selection",
    "params": {
        "image_path": str(image_path),
        "alpha_mask_path": str(alpha_path),
        "region_mask_path": str(region_path),
        "region_bbox": [12, 8, 73, 81],
        "model_id": "birefnet_hr_matting",
        "algorithm_version": "birefnet-hr-matting-lasso-v1",
        "fingerprint": "birefnet-smoke",
        "canvas_width": width,
        "canvas_height": height,
        "model_cache_dir": str(work / "models"),
        "runtime_cache_dir": str(work / "runtime"),
    },
}

worker = root / "worker" / "smart_select_worker.py"
completed = subprocess.run(
    [sys.executable, str(worker)],
    input=json.dumps(request) + "\n",
    text=True,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    check=False,
)
if completed.returncode != 0:
    raise SystemExit(completed.returncode)

messages = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
if not messages or "result" not in messages[-1]:
    raise SystemExit(f"Worker did not return result: {messages!r}")

mask_path = Path(messages[-1]["result"]["selection_mask_path"])
data = mask_path.read_bytes()
if not data.startswith(b"\x93NUMPY"):
    raise SystemExit("selection_mask.npy is not a numpy file")
payload = data[10 + struct.unpack("<H", data[8:10])[0] :]
nonzero = sum(value > 0 for value in payload)
soft = sum(0 < value < 255 for value in payload)
if nonzero <= 0:
    raise SystemExit("selection mask is empty")
print(f"BiRefNet smoke OK: nonzero={nonzero} soft={soft} mask={mask_path}")
PY
