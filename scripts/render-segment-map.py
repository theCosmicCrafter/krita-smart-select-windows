#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import struct
from pathlib import Path


def read_npy_u16(path: Path) -> tuple[int, int, list[int]]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise SystemExit(f"Not a NumPy .npy file: {path}")

    major = data[6]
    if major == 1:
        header_offset = 10
        header_length = struct.unpack("<H", data[8:10])[0]
    elif major == 2:
        header_offset = 12
        header_length = struct.unpack("<I", data[8:12])[0]
    else:
        raise SystemExit(f"Unsupported .npy version: {major}")

    header_end = header_offset + header_length
    header = ast.literal_eval(data[header_offset:header_end].decode("latin1"))
    if header.get("descr") != "<u2":
        raise SystemExit(f"Expected <u2 segment map, got {header.get('descr')!r}")
    if header.get("fortran_order"):
        raise SystemExit("Fortran-order arrays are not supported")

    shape = header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise SystemExit(f"Expected 2D segment map, got shape={shape!r}")
    height, width = int(shape[0]), int(shape[1])
    payload = data[header_end:]
    expected = width * height * 2
    if len(payload) < expected:
        raise SystemExit("Truncated segment map payload")

    values = list(struct.unpack("<" + "H" * (width * height), payload[:expected]))
    return width, height, values


def color_for_id(segment_id: int) -> tuple[int, int, int]:
    if segment_id == 0:
        return 0, 0, 0
    value = segment_id * 0x45D9F3B
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def render_ppm(
    output_path: Path,
    width: int,
    height: int,
    values: list[int],
    selected_segment_id: int | None,
) -> None:
    pixels = bytearray()
    for value in values:
        if selected_segment_id is not None:
            pixels.extend((255, 255, 255) if value == selected_segment_id else (0, 0, 0))
        else:
            pixels.extend(color_for_id(value))

    output_path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Smart Select segment_map.npy to PPM.")
    parser.add_argument("segment_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-id", type=int)
    args = parser.parse_args()

    width, height, values = read_npy_u16(args.segment_map)
    render_ppm(args.output, width, height, values, args.segment_id)
    segment_count = len({value for value in values if value > 0})
    print(f"wrote {args.output} ({width}x{height}, segments={segment_count})")


if __name__ == "__main__":
    main()
