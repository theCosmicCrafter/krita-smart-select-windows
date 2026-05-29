#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import threading
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

import image_preprocessing  # noqa: E402
import mam_refiner  # noqa: E402
import region_selection  # noqa: E402
import runtime_env  # noqa: E402
import sam2_inference  # noqa: E402
import segment_map_builder  # noqa: E402
import selection_mask_builder  # noqa: E402
import trimap_builder  # noqa: E402
from job_runner import JobContext, JobRunner  # noqa: E402
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
from observability import log_event  # noqa: E402
from refiner_cache import (  # noqa: E402
    RefinerCache,
    RefinerCacheDirMissing,
    RefinerCacheError,
    RefinerDownloadCancelled,
    RefinerDownloadError,
    RefinerUnavailable,
    RefinerVerificationError,
)
from refiner_registry import (  # noqa: E402
    RefinerRegistryError,
    RefinerSpec,
    default_refiner_id,
    get_refiner_spec,
)
from runtime_diagnostics import collect_runtime_diagnostics  # noqa: E402


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


class MaskStats(TypedDict):
    nonzero_pixels: int
    full_pixels: int
    soft_pixels: int
    alpha_sum: int
    max_alpha: int
    mean_nonzero_alpha: float


class MaskDeltaStats(TypedDict):
    added_vs_prior_pixels: int
    removed_vs_prior_pixels: int
    overlap_pixels: int
    outside_region_pixels: int


class PrepareSegmentsParams(TypedDict, total=False):
    image_path: str
    alpha_mask_path: str
    model_id: str
    algorithm_version: str
    fingerprint: str
    canvas_width: int
    canvas_height: int
    model_cache_dir: str
    runtime_cache_dir: str


class PrepareSelectionParams(PrepareSegmentsParams, total=False):
    region_mask_path: str
    region_bbox: list[int]
    refiner_id: str
    refiner_cache_dir: str


JsonObject = dict[str, Any]

cancelled_jobs: set[int] = set()
_emit_lock = threading.Lock()
_job_runner: JobRunner | None = None


def emit(message: JsonObject) -> None:
    with _emit_lock:
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


def error_type_for_runtime_error(error: runtime_env.RuntimeEnvError) -> str:
    if isinstance(error, runtime_env.RuntimeCacheDirMissing):
        return "RuntimeCacheDirMissing"
    if isinstance(error, runtime_env.RuntimeBootstrapError):
        return "RuntimeBootstrapError"
    if isinstance(error, runtime_env.RuntimeDependencyError):
        return "RuntimeDependencyError"
    if isinstance(error, runtime_env.RuntimeValidationError):
        return "RuntimeValidationError"
    if isinstance(error, runtime_env.RefinerRuntimeValidationError):
        return "RefinerRuntimeValidationError"
    return error.__class__.__name__


def error_type_for_inference_error(error: sam2_inference.Sam2InferenceError) -> str:
    if isinstance(error, sam2_inference.InferenceDeviceError):
        return "InferenceDeviceError"
    if isinstance(error, sam2_inference.InferenceOutOfMemory):
        return "InferenceOutOfMemory"
    if isinstance(error, sam2_inference.InferenceFailed):
        return "InferenceFailed"
    return error.__class__.__name__


def error_type_for_refiner_cache_error(error: RefinerCacheError) -> str:
    if isinstance(error, RefinerCacheDirMissing):
        return "RefinerCacheDirMissing"
    if isinstance(error, RefinerDownloadError):
        return "RefinerDownloadError"
    if isinstance(error, RefinerVerificationError):
        return "RefinerVerificationError"
    if isinstance(error, RefinerDownloadCancelled):
        return "RefinerDownloadCancelled"
    if isinstance(error, RefinerUnavailable):
        return "RefinerUnavailable"
    return error.__class__.__name__


def error_type_for_mam_error(error: mam_refiner.MamRefinerError) -> str:
    if isinstance(error, mam_refiner.MattingOutOfMemory):
        return "MattingOutOfMemory"
    if isinstance(error, mam_refiner.MattingFailed):
        return "MattingFailed"
    return error.__class__.__name__


def model_result_payload(spec: ModelSpec, cache_status_actual_bytes: int | None) -> JsonObject:
    return {
        "id": spec.id,
        "title": spec.title,
        "cache_state": "verified",
        "checkpoint_bytes": cache_status_actual_bytes,
    }


def refiner_result_payload(
    spec: RefinerSpec,
    result: mam_refiner.MamRefinementResult,
) -> JsonObject:
    return mam_refiner.runtime_result_payload(result, spec)


def crop_mask_prompt_box(
    mask: bytes, crop_bbox: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    width = crop_bbox[2]
    height = crop_bbox[3]
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            if mask[row_offset + x] == 0:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return (0.0, 0.0, float(max(1, width)), float(max(1, height)))
    padding = max(2, round(max(max_x - min_x + 1, max_y - min_y + 1) * 0.03))
    x0 = max(0, min_x - padding)
    y0 = max(0, min_y - padding)
    x1 = min(width, max_x + 1 + padding)
    y1 = min(height, max_y + 1 + padding)
    return (float(x0), float(y0), float(max(x0 + 1, x1)), float(max(y0 + 1, y1)))


def is_job_cancelled(job_id: int, context: JobContext | None) -> bool:
    return job_id in cancelled_jobs or (context is not None and context.is_cancelled())


def handle_prepare_segments(
    job_id: int, params: Mapping[str, Any], context: JobContext | None = None
) -> None:
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
    if is_job_cancelled(job_id, context):
        return

    image_path = params.get("image_path")
    if isinstance(image_path, str) and image_path and not Path(image_path).exists():
        emit_error(job_id, "InvalidInput", "Snapshot image not found")
        return
    if model_spec.id != "mock" and (not isinstance(image_path, str) or not image_path):
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
    if model_spec.id != "mock" and (not isinstance(alpha_mask_path, str) or not alpha_mask_path):
        emit_error(job_id, "InvalidInput", "Alpha mask not found")
        return

    runtime_ready: runtime_env.RuntimeReady | None = None
    if model_spec.id != "mock":
        runtime_cache_dir_value = params.get("runtime_cache_dir")
        if not isinstance(runtime_cache_dir_value, str) or not runtime_cache_dir_value:
            emit_error(job_id, "RuntimeCacheDirMissing", "runtime cache is not writable")
            return

        def runtime_progress(
            stage: str, progress: float | None, extra: Mapping[str, object]
        ) -> None:
            emit_progress_extra(job_id, stage, progress, extra)

        try:
            runtime_ready = runtime_env.ensure_runtime(
                Path(runtime_cache_dir_value), runtime_progress, context
            )
        except runtime_env.RuntimeEnvError as exc:
            emit_error(job_id, error_type_for_runtime_error(exc), str(exc))
            return

        if is_job_cancelled(job_id, context):
            return

    cache_status_actual_bytes: int | None = None
    checkpoint_path: Path | None = None
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
                lambda: is_job_cancelled(job_id, context),
            )
        except ModelDownloadCancelled:
            return
        except ModelCacheError as exc:
            emit_error(job_id, error_type_for_model_cache_error(exc), str(exc))
            return

        emit_progress(job_id, "verifying_model")
        cache_status_actual_bytes = cache_status.actual_bytes
        checkpoint_path = cache_status.checkpoint_path
        if is_job_cancelled(job_id, context):
            return

    emit_progress(job_id, "reading_image")
    time.sleep(0.03)
    if is_job_cancelled(job_id, context):
        return

    runtime_payload: JsonObject | None = None
    if model_spec.id == "mock":
        emit_progress(job_id, "generating_segments", 0.42)
        segment_map, metadata = fake_segments(width, height)
        if alpha_mask is not None:
            apply_alpha_mask(segment_map, alpha_mask)
            metadata = metadata_from_segment_map(segment_map, width, height)
    else:
        if checkpoint_path is None or runtime_ready is None:
            emit_error(job_id, "InferenceFailed", "SAM2 inference failed")
            return
        assert isinstance(image_path, str)
        assert isinstance(alpha_mask_path, str)
        emit_progress(job_id, "preprocessing_image")
        try:
            preprocessed = image_preprocessing.load_preprocessed_image(
                image_path,
                alpha_mask_path,
                width,
                height,
                read_npy_uint8_2d,
            )
        except (OSError, ValueError) as exc:
            emit_error(job_id, "InvalidInput", str(exc))
            return
        if preprocessed.resized:
            log_event(
                "image.preprocess.resized",
                component="worker",
                **{
                    "image.width": width,
                    "image.height": height,
                    "inference.image.width": preprocessed.inference_width,
                    "inference.image.height": preprocessed.inference_height,
                },
            )
        if is_job_cancelled(job_id, context):
            return

        emit_progress(job_id, "loading_model")
        log_event(
            "sam.generate.start",
            component="worker",
            **{"job.id": job_id, "job.type": "legacy_prepare_segments", "model.id": model_spec.id},
        )
        emit_progress(job_id, "running_sam")
        try:
            inference_result = sam2_inference.generate_masks(
                model_spec, checkpoint_path, preprocessed.image_array
            )
        except sam2_inference.Sam2InferenceError as exc:
            emit_error(job_id, error_type_for_inference_error(exc), str(exc))
            return
        runtime_payload = sam2_inference.runtime_result_payload(inference_result)
        if is_job_cancelled(job_id, context):
            return

        emit_progress(job_id, "building_segment_map")
        try:
            build_result = segment_map_builder.build_segment_map_from_masks(
                inference_result.masks,
                canvas_width=preprocessed.canvas_width,
                canvas_height=preprocessed.canvas_height,
                inference_width=preprocessed.inference_width,
                inference_height=preprocessed.inference_height,
                alpha_mask=preprocessed.alpha_mask,
            )
        except ValueError as exc:
            emit_error(job_id, "InferenceFailed", str(exc))
            return
        segment_map = build_result.segment_map
        metadata = build_result.metadata
        log_event(
            "segment_map.result",
            component="worker",
            **{
                "job.id": job_id,
                "job.type": "legacy_prepare_segments",
                "segments.count": len(metadata),
                "segment_map.nonzero_pixels": sum(1 for value in segment_map if value > 0),
            },
        )

    time.sleep(0.03)
    if is_job_cancelled(job_id, context):
        return

    emit_progress(job_id, "building_hover_index")
    temp_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-"))
    segment_map_path = temp_dir / "segment_map.npy"
    segments_path = temp_dir / "segments.npz"

    try:
        write_npy(segment_map_path, "<u2", (height, width), pack_uint16(segment_map))
        write_segments_npz(segments_path, metadata)
        if is_job_cancelled(job_id, context):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    result: JsonObject = {
        "segment_map_path": str(segment_map_path),
        "segments_path": str(segments_path),
        "metadata": metadata,
        "model": model_result_payload(model_spec, cache_status_actual_bytes),
    }
    if runtime_payload is not None:
        result["runtime"] = runtime_payload

    emit(
        {
            "id": job_id,
            "result": result,
        }
    )


def handle_prepare_selection(
    job_id: int, params: Mapping[str, Any], context: JobContext | None = None
) -> None:
    width_value = params.get("canvas_width")
    height_value = params.get("canvas_height")
    try:
        width = int(width_value) if width_value is not None else 0
        height = int(height_value) if height_value is not None else 0
    except (TypeError, ValueError):
        emit_error(job_id, "InvalidInput", "Canvas size is invalid")
        return

    if width <= 0 or height <= 0:
        emit_error(job_id, "InvalidInput", "Canvas size is invalid")
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
    if is_job_cancelled(job_id, context):
        return

    image_path = params.get("image_path")
    alpha_mask_path = params.get("alpha_mask_path")
    region_mask_path = params.get("region_mask_path")
    region_bbox_value = params.get("region_bbox")
    if not isinstance(image_path, str) or not image_path or not Path(image_path).exists():
        emit_error(job_id, "InvalidInput", "Snapshot image not found")
        return
    if not isinstance(alpha_mask_path, str) or not alpha_mask_path:
        emit_error(job_id, "InvalidInput", "Alpha mask not found")
        return
    if not isinstance(region_mask_path, str) or not region_mask_path:
        emit_error(job_id, "InvalidInput", "Region mask not found")
        return
    if not isinstance(region_bbox_value, list):
        emit_error(job_id, "InvalidInput", "Region bbox is invalid")
        return

    log_event(
        "selection.start",
        component="worker",
        **{"job.id": job_id, "job.type": "lasso_selection", "model.id": model_spec.id},
    )

    runtime_ready: runtime_env.RuntimeReady | None = None
    if model_spec.id != "mock":
        runtime_cache_dir_value = params.get("runtime_cache_dir")
        if not isinstance(runtime_cache_dir_value, str) or not runtime_cache_dir_value:
            emit_error(job_id, "RuntimeCacheDirMissing", "runtime cache is not writable")
            return

        def runtime_progress(
            stage: str, progress: float | None, extra: Mapping[str, object]
        ) -> None:
            emit_progress_extra(job_id, stage, progress, extra)

        try:
            runtime_ready = runtime_env.ensure_runtime(
                Path(runtime_cache_dir_value), runtime_progress, context
            )
        except runtime_env.RuntimeEnvError as exc:
            emit_error(job_id, error_type_for_runtime_error(exc), str(exc))
            return
        if is_job_cancelled(job_id, context):
            return

    cache_status_actual_bytes: int | None = None
    checkpoint_path: Path | None = None
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
                lambda: is_job_cancelled(job_id, context),
            )
        except ModelDownloadCancelled:
            return
        except ModelCacheError as exc:
            emit_error(job_id, error_type_for_model_cache_error(exc), str(exc))
            return

        emit_progress(job_id, "verifying_model")
        cache_status_actual_bytes = cache_status.actual_bytes
        checkpoint_path = cache_status.checkpoint_path
        if is_job_cancelled(job_id, context):
            return

    emit_progress(job_id, "reading_image")
    if is_job_cancelled(job_id, context):
        return

    runtime_payload: JsonObject | None = None
    refiner_payload: JsonObject | None = None
    try:
        selection_input = region_selection.load_region_selection_input(
            image_path,
            alpha_mask_path,
            region_mask_path,
            width,
            height,
            region_bbox_value,
            read_npy_uint8_2d,
        )
    except (OSError, ValueError) as exc:
        emit_error(job_id, "InvalidInput", str(exc))
        return

    log_event(
        "selection.prompt.result",
        component="worker",
        **{
            "job.id": job_id,
            "job.type": "lasso_selection",
            "region.bbox": selection_input.region_bbox,
            "region.positive_points.count": sum(
                1 for label in selection_input.point_labels if label == 1
            ),
            "region.negative_points.count": sum(
                1 for label in selection_input.point_labels if label == 0
            ),
        },
    )

    if is_job_cancelled(job_id, context):
        return

    emit_progress(job_id, "preprocessing_image")
    if model_spec.id == "mock":
        try:
            build_result = selection_mask_builder.build_selection_mask(
                [selection_input.crop_region_mask], [1.0], selection_input
            )
        except selection_mask_builder.EmptySelectionError as exc:
            emit_error(job_id, "EmptySelection", str(exc))
            return
    else:
        if checkpoint_path is None or runtime_ready is None:
            emit_error(job_id, "InferenceFailed", "SAM2 inference failed")
            return
        emit_progress(job_id, "loading_model")
        log_event(
            "sam.predict.start",
            component="worker",
            **{"job.id": job_id, "job.type": "lasso_selection", "model.id": model_spec.id},
        )
        emit_progress(job_id, "running_sam")
        try:
            inference_result = sam2_inference.predict_selection(
                model_spec,
                checkpoint_path,
                selection_input.crop_image_array,
                selection_input.prompt_box_xyxy,
                selection_input.point_coords,
                selection_input.point_labels,
            )
        except sam2_inference.Sam2InferenceError as exc:
            emit_error(job_id, error_type_for_inference_error(exc), str(exc))
            return
        runtime_payload = sam2_inference.runtime_prompted_result_payload(inference_result)
        if is_job_cancelled(job_id, context):
            return

        emit_progress(job_id, "building_selection_mask")
        try:
            prior_result = selection_mask_builder.build_sam_prior(
                inference_result.masks, inference_result.scores, selection_input
            )
        except selection_mask_builder.EmptySelectionError as exc:
            emit_error(job_id, "EmptySelection", str(exc))
            return
        except ValueError as exc:
            emit_error(job_id, "InferenceFailed", str(exc))
            return
        try:
            trimap = trimap_builder.build_trimap_from_prior(
                prior_result.crop_prior_mask,
                selection_input.crop_region_mask,
                selection_input.crop_alpha_mask,
                selection_input.crop_bbox[2],
                selection_input.crop_bbox[3],
            )
        except ValueError as exc:
            emit_error(job_id, "MattingFailed", str(exc))
            return

        emit_progress(job_id, "resolving_refiner")
        refiner_id_value = params.get("refiner_id", default_refiner_id())
        refiner_id = (
            refiner_id_value
            if isinstance(refiner_id_value, str) and refiner_id_value
            else default_refiner_id()
        )
        try:
            refiner_spec = get_refiner_spec(refiner_id)
        except RefinerRegistryError:
            emit_error(job_id, "RefinerUnavailable", "MAM checkpoint is not available")
            return
        if is_job_cancelled(job_id, context):
            return

        refiner_cache_dir_value = params.get("refiner_cache_dir")
        if not isinstance(refiner_cache_dir_value, str) or not refiner_cache_dir_value:
            emit_error(job_id, "RefinerCacheDirMissing", "edge refiner cache is not writable")
            return

        emit_progress(job_id, "checking_refiner_cache")
        refiner_cache = RefinerCache(Path(refiner_cache_dir_value))

        def refiner_progress(
            stage: str, progress: float | None, extra: Mapping[str, object]
        ) -> None:
            emit_progress_extra(job_id, stage, progress, extra)

        try:
            refiner_cache_status = refiner_cache.ensure_downloaded(
                refiner_spec,
                refiner_progress,
                lambda: is_job_cancelled(job_id, context),
            )
        except RefinerDownloadCancelled:
            return
        except RefinerCacheError as exc:
            emit_error(job_id, error_type_for_refiner_cache_error(exc), str(exc))
            return
        emit_progress(job_id, "verifying_refiner")
        if is_job_cancelled(job_id, context):
            return

        try:
            source_ready = runtime_env.ensure_mam_source(runtime_ready, refiner_progress, context)
        except runtime_env.RuntimeEnvError as exc:
            if isinstance(exc, runtime_env.RefinerRuntimeValidationError):
                mam_result = mam_refiner.fallback_from_sam_prior(
                    refiner_spec, prior_result.crop_prior_mask, error_type_for_runtime_error(exc)
                )
                emit_progress(job_id, "building_alpha_mask")
                try:
                    build_result = selection_mask_builder.build_selection_mask_from_crop_alpha(
                        mam_result.crop_alpha,
                        selection_input,
                        prior_result.best_score,
                        fallback_used=True,
                    )
                except selection_mask_builder.EmptySelectionError as empty_exc:
                    emit_error(job_id, "EmptySelection", str(empty_exc))
                    return
                refiner_payload = refiner_result_payload(refiner_spec, mam_result)
                _write_debug_exports(
                    selection_input,
                    prior_result.crop_prior_mask,
                    mam_result.crop_alpha,
                    build_result.selection_mask,
                    build_result.metadata,
                    trimap,
                    mam_result,
                )
                source_ready = None
            else:
                emit_error(job_id, error_type_for_runtime_error(exc), str(exc))
                return
        if source_ready is None:
            if is_job_cancelled(job_id, context):
                return
        elif is_job_cancelled(job_id, context):
            return
        if source_ready is None:
            pass
        else:
            emit_progress(job_id, "loading_refiner")
            emit_progress(job_id, "running_mam")
            mam_input = mam_refiner.MamRefinementInput(
                crop_rgb=selection_input.crop_image_array,
                crop_alpha_mask=selection_input.crop_alpha_mask,
                crop_region_mask=selection_input.crop_region_mask,
                crop_sam_prior_mask=prior_result.crop_prior_mask,
                trimap=trimap,
                crop_bbox=selection_input.crop_bbox,
                prompt_box_xyxy=crop_mask_prompt_box(
                    prior_result.crop_prior_mask, selection_input.crop_bbox
                ),
                canvas_width=selection_input.canvas_width,
                canvas_height=selection_input.canvas_height,
                prior_fallback_used=prior_result.fallback_used,
                prior_region_coverage=prior_result.region_coverage,
            )
            try:
                mam_result = mam_refiner.refine_with_mam(
                    refiner_spec,
                    refiner_cache_status,
                    runtime_ready,
                    source_ready,
                    mam_input,
                )
            except mam_refiner.MamRefinerError as exc:
                if isinstance(exc, mam_refiner.MattingFailed | mam_refiner.MattingOutOfMemory):
                    mam_result = mam_refiner.fallback_from_sam_prior(
                        refiner_spec, prior_result.crop_prior_mask, error_type_for_mam_error(exc)
                    )
                else:
                    emit_error(job_id, error_type_for_mam_error(exc), str(exc))
                    return
            if is_job_cancelled(job_id, context):
                return

            emit_progress(job_id, "building_alpha_mask")
            try:
                build_result = selection_mask_builder.build_selection_mask_from_crop_alpha(
                    mam_result.crop_alpha,
                    selection_input,
                    prior_result.best_score,
                    fallback_used=prior_result.fallback_used or mam_result.fallback_used,
                )
            except selection_mask_builder.EmptySelectionError as exc:
                emit_error(job_id, "EmptySelection", str(exc))
                return
            except ValueError as exc:
                emit_error(job_id, "MattingFailed", str(exc))
                return
            refiner_payload = refiner_result_payload(refiner_spec, mam_result)
            _write_debug_exports(
                selection_input,
                prior_result.crop_prior_mask,
                mam_result.crop_alpha,
                build_result.selection_mask,
                build_result.metadata,
                trimap,
                mam_result,
            )

    if is_job_cancelled(job_id, context):
        return

    if model_spec.id == "mock":
        emit_progress(job_id, "building_selection_mask")

    temp_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-"))
    selection_mask_path = temp_dir / "selection_mask.npy"
    segments_path = temp_dir / "segments.npz"

    try:
        write_npy(selection_mask_path, "|u1", (height, width), build_result.selection_mask)
        write_segments_npz(segments_path, build_result.metadata)
        if is_job_cancelled(job_id, context):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    area = build_result.metadata[0]["area"] if build_result.metadata else 0
    bbox = build_result.metadata[0]["bbox"] if build_result.metadata else []
    log_event(
        "selection_mask.result",
        component="worker",
        **{
            "job.id": job_id,
            "job.type": "lasso_selection",
            "selection.area": area,
            "selection.bbox": bbox,
        },
    )

    result: JsonObject = {
        "selection_mask_path": str(selection_mask_path),
        "segments_path": str(segments_path),
        "metadata": build_result.metadata,
        "model": model_result_payload(model_spec, cache_status_actual_bytes),
    }
    if runtime_payload is not None:
        result["runtime"] = runtime_payload
    if refiner_payload is not None:
        result["refiner"] = refiner_payload

    emit({"id": job_id, "result": result})


def _write_debug_exports(
    selection_input: region_selection.RegionSelectionInput,
    sam_prior_mask: bytes,
    mam_alpha: bytes,
    final_alpha: bytes,
    metadata: Sequence[SegmentMetadata],
    trimap: trimap_builder.TrimapBuildResult | None = None,
    mam_result: mam_refiner.MamRefinementResult | None = None,
) -> None:
    if not os.environ.get("SMART_SELECT_KEEP_DEBUG_FILES"):
        return
    try:
        image_module = __import__("PIL.Image", fromlist=["Image"])
        crop_width = selection_input.crop_bbox[2]
        crop_height = selection_input.crop_bbox[3]
        debug_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-debug-")) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        mam_stats = _mask_stats(mam_alpha)
        prior_stats = _mask_stats(sam_prior_mask)
        final_stats = _mask_stats(final_alpha)
        region_stats = _mask_stats(selection_input.crop_region_mask)
        trimap_stats: dict[str, object] | None = None
        delta_stats = _mask_delta_stats(
            sam_prior_mask,
            mam_alpha,
            selection_input.crop_region_mask,
        )
        image_module.fromarray(selection_input.crop_image_array).save(debug_dir / "crop_rgb.png")
        _save_debug_mask(
            image_module,
            debug_dir / "crop_region_mask.png",
            selection_input.crop_region_mask,
            crop_width,
            crop_height,
        )
        _save_debug_mask(
            image_module,
            debug_dir / "crop_alpha_mask.png",
            selection_input.crop_alpha_mask,
            crop_width,
            crop_height,
        )
        _save_debug_mask(
            image_module, debug_dir / "sam_prior_mask.png", sam_prior_mask, crop_width, crop_height
        )
        _save_debug_mask(
            image_module, debug_dir / "mam_alpha.png", mam_alpha, crop_width, crop_height
        )
        if trimap is not None:
            _save_debug_mask(
                image_module,
                debug_dir / "trimap.png",
                trimap.trimap_mask,
                crop_width,
                crop_height,
            )
            _save_debug_mask(
                image_module,
                debug_dir / "trimap_core_mask.png",
                trimap.core_mask,
                crop_width,
                crop_height,
            )
            _save_debug_mask(
                image_module,
                debug_dir / "trimap_unknown_mask.png",
                trimap.unknown_mask,
                crop_width,
                crop_height,
            )
            _save_debug_mask(
                image_module,
                debug_dir / "trimap_support_mask.png",
                trimap.support_mask,
                crop_width,
                crop_height,
            )
            trimap_stats = {
                "core_area": trimap.core_area,
                "core_erode_radius": trimap.core_erode_radius,
                "prior_area": trimap.prior_area,
                "support_area": trimap.support_area,
                "support_dilate_radius": trimap.support_dilate_radius,
                "unknown_area": trimap.unknown_area,
            }
        _save_debug_mask(
            image_module,
            debug_dir / "final_alpha.png",
            final_alpha,
            selection_input.canvas_width,
            selection_input.canvas_height,
        )
        _save_delta_debug_mask(
            image_module,
            debug_dir / "mam_added_vs_prior.png",
            sam_prior_mask,
            mam_alpha,
            crop_width,
            crop_height,
            added=True,
        )
        _save_delta_debug_mask(
            image_module,
            debug_dir / "mam_removed_vs_prior.png",
            sam_prior_mask,
            mam_alpha,
            crop_width,
            crop_height,
            added=False,
        )
        (debug_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "debug": {
                        "composition": {
                            "max_mam_added_vs_prior_fraction": (
                                mam_refiner.MAX_MAM_ADDED_VS_PRIOR_FRACTION
                            ),
                            "max_mam_added_vs_prior_pixels": (
                                mam_refiner.MAX_MAM_ADDED_VS_PRIOR_PIXELS
                            ),
                            "max_mam_area_vs_prior_ratio": (
                                mam_refiner.MAX_MAM_AREA_VS_PRIOR_RATIO
                            ),
                            "max_mam_removed_vs_prior_fraction": (
                                mam_refiner.MAX_MAM_REMOVED_VS_PRIOR_FRACTION
                            ),
                            "max_mam_removed_vs_prior_pixels": (
                                mam_refiner.MAX_MAM_REMOVED_VS_PRIOR_PIXELS
                            ),
                            "min_mam_area_vs_prior_ratio": (
                                mam_refiner.MIN_MAM_AREA_VS_PRIOR_RATIO
                            ),
                            "os1_width": mam_refiner.OS1_WIDTH,
                            "os4_width": mam_refiner.OS4_WIDTH,
                            "os8_width": mam_refiner.OS8_WIDTH,
                        },
                        "crop_bbox": selection_input.crop_bbox,
                        "final_alpha": final_stats,
                        "mam_alpha": mam_stats,
                        "mam_result": {
                            "fallback_used": mam_result.fallback_used if mam_result else None,
                            "rejected": mam_result.rejected if mam_result else None,
                            "rejection_reason": (
                                mam_result.rejection_reason if mam_result else None
                            ),
                        },
                        "mam_vs_sam_prior": delta_stats,
                        "region_mask": region_stats,
                        "sam_prior_mask": prior_stats,
                        "trimap": trimap_stats,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        log_event(
            "selection.debug_export.metrics",
            component="worker",
            **{
                "selection.area": final_stats["nonzero_pixels"],
                "selection.alpha_sum": final_stats["alpha_sum"],
                "selection.soft_pixels": final_stats["soft_pixels"],
                "selection.prior.area": prior_stats["nonzero_pixels"],
                "selection.region.area": region_stats["nonzero_pixels"],
                "refinement.added_vs_prior_pixels": delta_stats["added_vs_prior_pixels"],
                "refinement.removed_vs_prior_pixels": delta_stats["removed_vs_prior_pixels"],
                "refinement.outside_region_pixels": delta_stats["outside_region_pixels"],
                "refinement.mam_full_pixels": mam_stats["full_pixels"],
                "refinement.mam_soft_pixels": mam_stats["soft_pixels"],
                "refinement.rejected": mam_result.rejected if mam_result else None,
                "trimap.core.area": trimap.core_area if trimap else None,
                "trimap.support.area": trimap.support_area if trimap else None,
                "trimap.unknown.area": trimap.unknown_area if trimap else None,
            },
        )
        log_event(
            "selection.debug_export.result",
            component="worker",
            **{"file.path": str(debug_dir)},
        )
    except Exception as exc:
        log_event(
            "selection.debug_export.failed",
            severity="warning",
            component="worker",
            **{"error.type": exc.__class__.__name__, "error.reason": str(exc)},
        )


def _save_debug_mask(image_module: Any, path: Path, data: bytes, width: int, height: int) -> None:
    image_module.frombytes("L", (width, height), data).save(path)


def _save_delta_debug_mask(
    image_module: Any,
    path: Path,
    prior: bytes,
    candidate: bytes,
    width: int,
    height: int,
    *,
    added: bool,
) -> None:
    if len(prior) != len(candidate):
        raise ValueError("debug mask sizes differ")
    output = bytearray(len(prior))
    for index, (prior_value, candidate_value) in enumerate(zip(prior, candidate, strict=True)):
        if added:
            output[index] = 255 if candidate_value > 0 and prior_value == 0 else 0
        else:
            output[index] = 255 if prior_value > 0 and candidate_value == 0 else 0
    image_module.frombytes("L", (width, height), bytes(output)).save(path)


def _mask_stats(data: bytes) -> MaskStats:
    nonzero_pixels = 0
    full_pixels = 0
    soft_pixels = 0
    alpha_sum = 0
    max_alpha = 0
    for value in data:
        alpha_sum += value
        if value > max_alpha:
            max_alpha = value
        if value == 0:
            continue
        nonzero_pixels += 1
        if value == 255:
            full_pixels += 1
        else:
            soft_pixels += 1
    return {
        "nonzero_pixels": nonzero_pixels,
        "full_pixels": full_pixels,
        "soft_pixels": soft_pixels,
        "alpha_sum": alpha_sum,
        "max_alpha": max_alpha,
        "mean_nonzero_alpha": round(alpha_sum / nonzero_pixels, 4) if nonzero_pixels else 0.0,
    }


def _mask_delta_stats(prior: bytes, candidate: bytes, region: bytes) -> MaskDeltaStats:
    if len(prior) != len(candidate) or len(prior) != len(region):
        raise ValueError("debug mask sizes differ")
    added_vs_prior = 0
    removed_vs_prior = 0
    overlap = 0
    outside_region = 0
    for prior_value, candidate_value, region_value in zip(prior, candidate, region, strict=True):
        prior_on = prior_value > 0
        candidate_on = candidate_value > 0
        if candidate_on and not prior_on:
            added_vs_prior += 1
        if prior_on and not candidate_on:
            removed_vs_prior += 1
        if prior_on and candidate_on:
            overlap += 1
        if candidate_on and region_value == 0:
            outside_region += 1
    return {
        "added_vs_prior_pixels": added_vs_prior,
        "removed_vs_prior_pixels": removed_vs_prior,
        "overlap_pixels": overlap,
        "outside_region_pixels": outside_region,
    }


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
        runner = job_runner()
        runner.start(job_id, {"_method": "prepare_segments", **params_value})
    elif method == "prepare_selection":
        runner = job_runner()
        runner.start(job_id, {"_method": "prepare_selection", **params_value})
    elif method == "cancel":
        cancelled_job_id = int(params_value.get("job_id") or 0)
        cancelled_jobs.add(cancelled_job_id)
        runner = job_runner()
        runner.cancel(cancelled_job_id)
        emit({"id": job_id, "result": {"cancelled": True}})
    elif method == "diagnostics":
        model_cache_dir_value = params_value.get("model_cache_dir")
        runtime_cache_dir_value = params_value.get("runtime_cache_dir")
        refiner_cache_dir_value = params_value.get("refiner_cache_dir")
        model_cache_dir = (
            Path(model_cache_dir_value)
            if isinstance(model_cache_dir_value, str) and model_cache_dir_value
            else None
        )
        runtime_cache_dir = (
            Path(runtime_cache_dir_value)
            if isinstance(runtime_cache_dir_value, str) and runtime_cache_dir_value
            else None
        )
        refiner_cache_dir = (
            Path(refiner_cache_dir_value)
            if isinstance(refiner_cache_dir_value, str) and refiner_cache_dir_value
            else None
        )
        emit(
            {
                "id": job_id,
                "result": collect_runtime_diagnostics(
                    model_cache_dir, runtime_cache_dir, refiner_cache_dir
                ),
            }
        )
    else:
        emit_error(job_id, "UnknownMethod", f"Unsupported method: {method}")


def job_runner() -> JobRunner:
    global _job_runner
    if _job_runner is None:
        _job_runner = JobRunner(handle_prepare_job, emit_error)
    return _job_runner


def handle_prepare_job(context: JobContext, params: Mapping[str, Any]) -> None:
    method = params.get("_method")
    payload = {key: value for key, value in params.items() if key != "_method"}
    if method == "prepare_selection":
        handle_prepare_selection(context.job_id, payload, context)
        return
    handle_prepare_segments(context.job_id, payload, context)


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
    runner = _job_runner
    if runner is not None:
        runner.wait_for_idle()


if __name__ == "__main__":
    main()
