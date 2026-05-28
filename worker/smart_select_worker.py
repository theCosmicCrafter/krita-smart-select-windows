#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
import time
import traceback
import zipfile
from array import array
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from model_cache import (  # noqa: E402
    ModelCache,
    ModelCacheDirMissing,
    ModelCacheError,
    ModelDownloadCancelled,
    ModelDownloadError,
    ModelVerificationError,
)
from model_registry import (  # noqa: E402
    ModelRegistryError,
    ModelSpec,
    default_model_id,
    get_model_spec,
)
from runtime_diagnostics import collect_runtime_diagnostics  # noqa: E402


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


class PrepareSegmentsParams(TypedDict, total=False):
    image_path: str
    alpha_mask_path: str
    model_id: str
    algorithm_version: str
    fingerprint: str
    canvas_width: int
    canvas_height: int
    model_cache_dir: str


JsonObject = dict[str, Any]

cancelled_jobs: set[int] = set()


def emit(message: JsonObject) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def emit_progress(job_id: int, stage: str, progress: float | None = None) -> None:
    emit({"id": job_id, "event": "progress", "stage": stage, "progress": progress})


def emit_progress_extra(
    job_id: int,
    stage: str,
    progress: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    message: JsonObject = {"id": job_id, "event": "progress", "stage": stage, "progress": progress}
    if extra is not None:
        message.update(extra)
    emit(message)


def emit_error(job_id: int, error_type: str, message: str) -> None:
    emit({"id": job_id, "error": {"type": error_type, "message": message}})


def metadata_from_segment_map(
    segment_map: Sequence[int], width: int, height: int
) -> list[SegmentMetadata]:
    ids = sorted({int(value) for value in segment_map if value > 0})
    metadata: list[SegmentMetadata] = []
    for segment_id in ids:
        xs: list[int] = []
        ys: list[int] = []
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
    return metadata


def fake_segments(width: int, height: int) -> tuple[array[int], list[SegmentMetadata]]:
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

    return segment_map, metadata_from_segment_map(segment_map, width, height)


def npy_bytes(descr: str, shape: Sequence[int], data: bytes) -> bytes:
    if len(shape) == 1:
        shape_text = f"({shape[0]},)"
    else:
        shape_text = "(" + ", ".join(str(value) for value in shape) + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data


def write_npy(path: str | Path, descr: str, shape: Sequence[int], data: bytes) -> None:
    Path(path).write_bytes(npy_bytes(descr, shape, data))


def read_npy_uint8_2d(path: str | Path) -> tuple[int, int, bytes]:
    data = Path(path).read_bytes()
    if len(data) < 16 or not data.startswith(b"\x93NUMPY"):
        raise ValueError("Not a NumPy .npy file")

    major = data[6]
    if major == 1:
        header_offset = 10
        header_length = struct.unpack("<H", data[8:10])[0]
    elif major == 2:
        header_offset = 12
        header_length = struct.unpack("<I", data[8:12])[0]
    else:
        raise ValueError("Unsupported .npy version")

    header_end = header_offset + header_length
    if len(data) < header_end:
        raise ValueError("Truncated .npy header")

    header = data[header_offset:header_end].decode("latin1")
    if "'descr': '|u1'" not in header and '"descr": "|u1"' not in header:
        raise ValueError("Expected uint8 alpha mask")
    if "'fortran_order': True" in header or '"fortran_order": true' in header:
        raise ValueError("Fortran-order .npy arrays are not supported")

    marker = "'shape': ("
    if marker not in header:
        raise ValueError("Could not parse .npy shape")
    shape_text = header.split(marker, 1)[1].split(")", 1)[0]
    shape = tuple(int(value.strip()) for value in shape_text.split(",") if value.strip())
    if len(shape) != 2:
        raise ValueError("Expected 2D alpha mask")

    height, width = shape
    payload = data[header_end:]
    expected_size = width * height
    if width <= 0 or height <= 0 or len(payload) < expected_size:
        raise ValueError("Truncated .npy alpha mask payload")

    return width, height, payload[:expected_size]


def apply_alpha_mask(segment_map: array[int], alpha_mask: bytes) -> None:
    if len(segment_map) != len(alpha_mask):
        raise ValueError("Alpha mask size does not match segment map")
    for index, alpha in enumerate(alpha_mask):
        if alpha == 0:
            segment_map[index] = 0


def pack_uint16(values: Iterable[int]) -> bytes:
    data = array("H", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def pack_int32(values: Sequence[int]) -> bytes:
    return struct.pack("<" + "i" * len(values), *values)


def pack_float64(values: Sequence[float]) -> bytes:
    return struct.pack("<" + "d" * len(values), *values)


def write_segments_npz(path: str | Path, metadata: Sequence[SegmentMetadata]) -> None:
    ids = [int(item["id"]) for item in metadata]
    bboxes = [int(value) for item in metadata for value in item["bbox"]]
    areas = [int(item["area"]) for item in metadata]
    scores = [float(item.get("score", 1.0)) for item in metadata]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ids.npy", npy_bytes("<u2", (len(ids),), pack_uint16(ids)))
        archive.writestr("bboxes.npy", npy_bytes("<i4", (len(ids), 4), pack_int32(bboxes)))
        archive.writestr("areas.npy", npy_bytes("<i4", (len(ids),), pack_int32(areas)))
        archive.writestr("scores.npy", npy_bytes("<f8", (len(ids),), pack_float64(scores)))


def error_type_for_model_cache_error(error: ModelCacheError) -> str:
    if isinstance(error, ModelCacheDirMissing):
        return "ModelCacheDirMissing"
    if isinstance(error, ModelDownloadError):
        return "ModelDownloadError"
    if isinstance(error, ModelVerificationError):
        return "ModelVerificationError"
    if isinstance(error, ModelDownloadCancelled):
        return "ModelDownloadCancelled"
    return error.__class__.__name__


def model_result_payload(spec: ModelSpec, cache_status_actual_bytes: int | None) -> JsonObject:
    return {
        "id": spec.id,
        "title": spec.title,
        "cache_state": "verified",
        "checkpoint_bytes": cache_status_actual_bytes,
    }


def handle_prepare_segments(job_id: int, params: Mapping[str, Any]) -> None:
    width_value = params.get("canvas_width", 512)
    height_value = params.get("canvas_height", 512)
    if width_value is None:
        width_value = 512
    if height_value is None:
        height_value = 512
    try:
        width = int(width_value)
        height = int(height_value)
    except (TypeError, ValueError):
        emit_error(job_id, "InvalidInput", "Canvas size is invalid")
        return

    if width <= 0 or height <= 0:
        emit_error(job_id, "InvalidInput", "Canvas size is invalid")
        return

    image_path = params.get("image_path")
    if isinstance(image_path, str) and image_path and not Path(image_path).exists():
        emit_error(job_id, "InvalidInput", "Snapshot image not found")
        return

    alpha_mask: bytes | None = None
    alpha_mask_path = params.get("alpha_mask_path")
    if isinstance(alpha_mask_path, str) and alpha_mask_path:
        try:
            mask_width, mask_height, alpha_mask = read_npy_uint8_2d(alpha_mask_path)
        except (OSError, ValueError) as exc:
            emit_error(job_id, "InvalidInput", str(exc))
            return
        if mask_width != width or mask_height != height:
            emit_error(job_id, "InvalidInput", "Alpha mask size does not match canvas size")
            return

    emit_progress(job_id, "resolving_model")
    model_id_value = params.get("model_id", default_model_id())
    model_id = (
        model_id_value if isinstance(model_id_value, str) and model_id_value else default_model_id()
    )
    try:
        model_spec = get_model_spec(model_id)
    except ModelRegistryError:
        emit_error(job_id, "UnknownModel", "unknown model")
        return
    if job_id in cancelled_jobs:
        return

    cache_status_actual_bytes: int | None = None
    if model_spec.requires_download:
        model_cache_dir_value = params.get("model_cache_dir")
        if not isinstance(model_cache_dir_value, str) or not model_cache_dir_value:
            emit_error(job_id, "ModelCacheDirMissing", "model cache is not writable")
            return

        emit_progress(job_id, "checking_model_cache")
        model_cache = ModelCache(Path(model_cache_dir_value))

        def progress_callback(
            stage: str, progress: float | None, extra: Mapping[str, object]
        ) -> None:
            emit_progress_extra(job_id, stage, progress, extra)

        try:
            cache_status = model_cache.ensure_downloaded(
                model_spec,
                progress_callback,
                lambda: job_id in cancelled_jobs,
            )
        except ModelDownloadCancelled:
            return
        except ModelCacheError as exc:
            emit_error(job_id, error_type_for_model_cache_error(exc), str(exc))
            return

        emit_progress(job_id, "verifying_model")
        cache_status_actual_bytes = cache_status.actual_bytes
        if job_id in cancelled_jobs:
            return

    emit_progress(job_id, "reading_image")
    time.sleep(0.03)
    if job_id in cancelled_jobs:
        return

    emit_progress(job_id, "generating_segments", 0.42)
    segment_map, metadata = fake_segments(width, height)
    if alpha_mask is not None:
        apply_alpha_mask(segment_map, alpha_mask)
        metadata = metadata_from_segment_map(segment_map, width, height)
    time.sleep(0.03)
    if job_id in cancelled_jobs:
        return

    emit_progress(job_id, "building_hover_index")
    temp_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-"))
    segment_map_path = temp_dir / "segment_map.npy"
    segments_path = temp_dir / "segments.npz"

    try:
        write_npy(segment_map_path, "<u2", (height, width), pack_uint16(segment_map))
        write_segments_npz(segments_path, metadata)
        if job_id in cancelled_jobs:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    emit(
        {
            "id": job_id,
            "result": {
                "segment_map_path": str(segment_map_path),
                "segments_path": str(segments_path),
                "metadata": metadata,
                "model": model_result_payload(model_spec, cache_status_actual_bytes),
            },
        }
    )


def handle_request(request: Mapping[str, Any]) -> None:
    job_id = int(request.get("id") or 0)
    method = request.get("method")
    params_value = request.get("params", {})
    if params_value is None:
        params_value = {}
    if not isinstance(params_value, dict):
        emit_error(job_id, "InvalidInput", "Request params must be an object")
        return

    if method == "prepare_segments":
        handle_prepare_segments(job_id, params_value)
    elif method == "cancel":
        cancelled_jobs.add(int(params_value.get("job_id") or 0))
        emit({"id": job_id, "result": {"cancelled": True}})
    elif method == "diagnostics":
        model_cache_dir_value = params_value.get("model_cache_dir")
        model_cache_dir = (
            Path(model_cache_dir_value)
            if isinstance(model_cache_dir_value, str) and model_cache_dir_value
            else None
        )
        emit({"id": job_id, "result": collect_runtime_diagnostics(model_cache_dir)})
    else:
        emit_error(job_id, "UnknownMethod", f"Unsupported method: {method}")


def main() -> None:
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
