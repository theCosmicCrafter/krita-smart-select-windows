#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import time
import traceback
import zipfile
from array import array
from pathlib import Path


cancelled_jobs = set()


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def emit_progress(job_id, stage, progress=None):
    emit({"id": job_id, "event": "progress", "stage": stage, "progress": progress})


def emit_error(job_id, error_type, message):
    emit({"id": job_id, "error": {"type": error_type, "message": message}})


def fake_segments(width, height):
    segment_map = array("H", [0]) * (width * height)

    top_x = max(0, width // 12)
    top_y = max(0, height // 12)
    top_w = max(8, width // 4)
    top_h = max(8, height // 4)
    for y in range(top_y, min(height, top_y + top_h)):
        for x in range(top_x, min(width, top_x + top_w)):
            segment_map[y * width + x] = 1

    cx = width // 2
    cy = height // 2
    rx = max(10, width // 8)
    ry = max(10, height // 8)
    for y in range(max(0, cy - ry), min(height, cy + ry)):
        for x in range(max(0, cx - rx), min(width, cx + rx)):
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            if dx * dx + dy * dy <= 1.0:
                segment_map[y * width + x] = 2

    br_x = max(0, width - width // 3)
    br_y = max(0, height - height // 3)
    br_w = max(8, width // 4)
    br_h = max(8, height // 4)
    for y in range(br_y, min(height, br_y + br_h)):
        for x in range(br_x, min(width, br_x + br_w)):
            segment_map[y * width + x] = 3

    metadata = []
    for segment_id in (1, 2, 3):
        xs = []
        ys = []
        for index, value in enumerate(segment_map):
            if value == segment_id:
                ys.append(index // width)
                xs.append(index % width)
        if not xs:
            continue
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(xs)
        y1 = max(ys)
        metadata.append(
            {
                "id": segment_id,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "area": len(xs),
                "score": 1.0,
            }
        )

    return segment_map, metadata


def npy_bytes(descr, shape, data):
    if len(shape) == 1:
        shape_text = f"({shape[0]},)"
    else:
        shape_text = "(" + ", ".join(str(value) for value in shape) + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data


def write_npy(path, descr, shape, data):
    Path(path).write_bytes(npy_bytes(descr, shape, data))


def pack_uint16(values):
    data = array("H", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def pack_int32(values):
    return struct.pack("<" + "i" * len(values), *values)


def pack_float64(values):
    return struct.pack("<" + "d" * len(values), *values)


def write_segments_npz(path, metadata):
    ids = [int(item["id"]) for item in metadata]
    bboxes = [int(value) for item in metadata for value in item["bbox"]]
    areas = [int(item["area"]) for item in metadata]
    scores = [float(item.get("score", 1.0)) for item in metadata]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ids.npy", npy_bytes("<u2", (len(ids),), pack_uint16(ids)))
        archive.writestr("bboxes.npy", npy_bytes("<i4", (len(ids), 4), pack_int32(bboxes)))
        archive.writestr("areas.npy", npy_bytes("<i4", (len(ids),), pack_int32(areas)))
        archive.writestr("scores.npy", npy_bytes("<f8", (len(ids),), pack_float64(scores)))


def handle_prepare_segments(job_id, params):
    width = int(params.get("canvas_width") or 512)
    height = int(params.get("canvas_height") or 512)
    if width <= 0 or height <= 0:
        emit_error(job_id, "InvalidInput", "Canvas size is invalid")
        return

    emit_progress(job_id, "loading_model")
    time.sleep(0.03)
    if job_id in cancelled_jobs:
        return

    emit_progress(job_id, "reading_image")
    time.sleep(0.03)
    if job_id in cancelled_jobs:
        return

    emit_progress(job_id, "generating_segments", 0.42)
    segment_map, metadata = fake_segments(width, height)
    time.sleep(0.03)
    if job_id in cancelled_jobs:
        return

    emit_progress(job_id, "building_hover_index")
    temp_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-"))
    segment_map_path = temp_dir / "segment_map.npy"
    segments_path = temp_dir / "segments.npz"

    write_npy(segment_map_path, "<u2", (height, width), pack_uint16(segment_map))
    write_segments_npz(segments_path, metadata)

    emit(
        {
            "id": job_id,
            "result": {
                "segment_map_path": str(segment_map_path),
                "segments_path": str(segments_path),
                "metadata": metadata,
            },
        }
    )


def handle_request(request):
    job_id = int(request.get("id") or 0)
    method = request.get("method")
    params = request.get("params") or {}

    if method == "prepare_segments":
        handle_prepare_segments(job_id, params)
    elif method == "cancel":
        cancelled_jobs.add(int(params.get("job_id") or 0))
        emit({"id": job_id, "result": {"cancelled": True}})
    else:
        emit_error(job_id, "UnknownMethod", f"Unsupported method: {method}")


def main():
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            handle_request(json.loads(line))
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            try:
                job_id = int(json.loads(line).get("id") or 0)
            except Exception:
                job_id = 0
            emit_error(job_id, exc.__class__.__name__, "Smart Select worker failed")


if __name__ == "__main__":
    main()
