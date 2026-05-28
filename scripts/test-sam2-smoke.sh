#!/usr/bin/env bash
set -euo pipefail

if [[ "${SMART_SELECT_RUN_SAM2_SMOKE:-}" != "1" ]]; then
  cat >&2 <<'EOF'
This smoke test may install PyTorch/SAM2 and download the SAM2.1 Large checkpoint.
Run explicitly with:
  SMART_SELECT_RUN_SAM2_SMOKE=1 ./scripts/test-sam2-smoke.sh
EOF
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CACHE_ROOT="${SMART_SELECT_CACHE_ROOT:-$HOME/.local/share/krita/smart_select}"
WORK_DIR="$(mktemp -d -t krita-smart-select-smoke-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

"$PYTHON" - "$ROOT_DIR" "$WORK_DIR" "$CACHE_ROOT" <<'PY'
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

root = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
width = 96
height = 96


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_png(path: Path) -> None:
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            inside = 24 <= x < 72 and 24 <= y < 72
            row.extend((40, 180, 80) if inside else (240, 240, 240))
        rows.append(bytes(row))
    raw = b"".join(rows)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def write_npy_uint8(path: Path, payload: bytes) -> None:
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d), }" % (height, width)
    encoded = header.encode("latin1")
    padding = 16 - ((10 + len(encoded) + 1) % 16)
    encoded += b" " * padding + b"\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + payload)


image_path = work_dir / "snapshot.png"
alpha_path = work_dir / "alpha.npy"
write_png(image_path)
write_npy_uint8(alpha_path, b"\xff" * (width * height))

request = {
    "id": 1,
    "method": "prepare_segments",
    "params": {
        "image_path": str(image_path),
        "alpha_mask_path": str(alpha_path),
        "model_id": "sam2.1_hiera_large",
        "algorithm_version": "sam2.1-auto-mask-v1",
        "fingerprint": "sam2-smoke",
        "canvas_width": width,
        "canvas_height": height,
        "model_cache_dir": str(cache_root / "models"),
        "runtime_cache_dir": str(cache_root / "runtime"),
    },
}

completed = subprocess.run(
    [sys.executable, str(root / "worker" / "smart_select_worker.py")],
    input=json.dumps(request) + "\n",
    text=True,
    capture_output=True,
    timeout=1800,
    check=False,
)
print(completed.stderr, file=sys.stderr, end="")
if completed.returncode != 0:
    raise SystemExit(completed.returncode)

messages = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
if not messages or "result" not in messages[-1]:
    print(completed.stdout, file=sys.stderr)
    raise SystemExit("worker did not return a result")

segment_map_path = Path(messages[-1]["result"]["segment_map_path"])
data = segment_map_path.read_bytes()
if b"\x01\x00" not in data and b"\x02\x00" not in data:
    raise SystemExit("segment map looks empty")

print(json.dumps(messages[-1]["result"]["runtime"], indent=2, sort_keys=True))
PY
