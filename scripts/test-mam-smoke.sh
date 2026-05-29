#!/usr/bin/env bash
set -euo pipefail

if [[ "${SMART_SELECT_RUN_MAM_SMOKE:-}" != "1" ]]; then
  echo "Set SMART_SELECT_RUN_MAM_SMOKE=1 to run the real MAM smoke test." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CACHE_ROOT="${SMART_SELECT_SMOKE_CACHE:-$HOME/.local/share/krita/smart_select}"
WORK_DIR="$(mktemp -d -t krita-smart-select-mam-smoke-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

"$PYTHON" - "$ROOT_DIR" "$WORK_DIR" "$CACHE_ROOT" <<'PY'
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
cache_root = Path(sys.argv[3])


def npy_bytes(descr: str, shape: tuple[int, int], payload: bytes) -> bytes:
    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header_text = str(header).replace("{", "{").replace("}", "}")
    header_bytes = header_text.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload


image_path = work_dir / "snapshot.png"
alpha_path = work_dir / "alpha.npy"
region_path = work_dir / "region.npy"
width, height = 96, 96

image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((24, 18, 76, 72), fill=(230, 180, 60), outline=(30, 30, 30), width=2)
draw.rectangle((44, 54, 54, 84), fill=(80, 80, 80))
image.save(image_path)

alpha = bytes([255] * (width * height))
region_image = Image.new("L", (width, height), 0)
ImageDraw.Draw(region_image).ellipse((16, 10, 84, 90), fill=255)
region = region_image.tobytes()

alpha_path.write_bytes(npy_bytes("|u1", (height, width), alpha))
region_path.write_bytes(npy_bytes("|u1", (height, width), region))

request = {
    "id": 1,
    "method": "prepare_selection",
    "params": {
        "image_path": str(image_path),
        "alpha_mask_path": str(alpha_path),
        "region_mask_path": str(region_path),
        "region_bbox": [16, 10, 68, 80],
        "model_id": "sam2.1_hiera_large",
        "refiner_id": "mam_sam_vith",
        "algorithm_version": "sam2.1-lasso-trimap-mam-v1",
        "fingerprint": "mam-smoke",
        "canvas_width": width,
        "canvas_height": height,
        "model_cache_dir": str(cache_root / "models"),
        "runtime_cache_dir": str(cache_root / "runtime"),
        "refiner_cache_dir": str(cache_root / "refiners"),
    },
}

completed = subprocess.run(
    [sys.executable, str(root / "worker" / "smart_select_worker.py")],
    input=json.dumps(request) + "\n",
    text=True,
    capture_output=True,
    check=False,
    timeout=1800,
)
if completed.returncode != 0:
    print(completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)

messages = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
if not messages or "result" not in messages[-1]:
    print(completed.stdout, file=sys.stderr)
    print(completed.stderr, file=sys.stderr)
    raise SystemExit("worker did not return a result")

mask_path = Path(messages[-1]["result"]["selection_mask_path"])
data = mask_path.read_bytes()
header_len = struct.unpack("<H", data[8:10])[0]
payload = data[10 + header_len :]
values = set(payload)
if not any(0 < value < 255 for value in values):
    raise SystemExit("MAM smoke did not produce soft alpha values")
print(mask_path)
PY
