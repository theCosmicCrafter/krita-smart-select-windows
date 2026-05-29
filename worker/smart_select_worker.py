#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NotRequired, TypedDict

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import alpha_selection_builder  # noqa: E402
import birefnet_inference  # noqa: E402
import region_selection  # noqa: E402
import runtime_env  # noqa: E402
from alpha_model_cache import (  # noqa: E402
    ModelCache,
    ModelCacheDirMissing,
    ModelCacheError,
    ModelCacheStatus,
    ModelDownloadCancelled,
    ModelDownloadError,
    ModelVerificationError,
)
from alpha_model_registry import (  # noqa: E402
    ModelRegistryError,
    ModelSpec,
    default_model_id,
    get_model_spec,
)
from job_runner import JobContext, JobRunner  # noqa: E402
from observability import log_event  # noqa: E402
from runtime_diagnostics import collect_runtime_diagnostics  # noqa: E402


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


class PrepareSelectionParams(TypedDict, total=False):
    image_path: str
    alpha_mask_path: str
    region_mask_path: str
    region_bbox: list[int]
    model_id: str
    algorithm_version: str
    fingerprint: str
    canvas_width: int
    canvas_height: int
    model_cache_dir: str
    runtime_cache_dir: str


JsonObject = dict[str, Any]

_emit_lock = threading.Lock()
_job_runner: JobRunner | None = None


def emit(message: JsonObject) -> None:
    with _emit_lock:
        sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def emit_progress(job_id: int, stage: str, progress: float | None = None) -> None:
    emit({"id": job_id, "event": "progress", "stage": stage, "progress": progress})


def emit_error(job_id: int, error_type: str, message: str) -> None:
    emit({"id": job_id, "error": {"type": error_type, "message": message}})


def npy_bytes(descr: str, shape: tuple[int, int], data: bytes) -> bytes:
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape[0]}, {shape[1]}), }}"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data


def write_npy(path: str | Path, descr: str, shape: tuple[int, int], data: bytes) -> None:
    Path(path).write_bytes(npy_bytes(descr, shape, data))


def read_npy_uint8_2d(path: str | Path) -> tuple[int, int, bytes]:
    data = Path(path).read_bytes()
    if len(data) < 16 or not data.startswith(b"\x93NUMPY"):
        raise ValueError("Invalid .npy file")
    major = data[6]
    if major != 1:
        raise ValueError("Unsupported .npy version")
    header_len = struct.unpack("<H", data[8:10])[0]
    header_start = 10
    header_end = header_start + header_len
    header = data[header_start:header_end].decode("latin1")
    if "'descr': '|u1'" not in header and '"descr": "|u1"' not in header:
        raise ValueError("Expected uint8 mask")
    if "'fortran_order': True" in header or '"fortran_order": true' in header:
        raise ValueError("Fortran-order .npy arrays are not supported")
    shape = _parse_shape(header)
    height, width = shape
    payload = data[header_end:]
    expected = width * height
    if len(payload) != expected:
        raise ValueError("Invalid uint8 mask payload size")
    return width, height, payload


def _parse_shape(header: str) -> tuple[int, int]:
    marker = "'shape':"
    index = header.find(marker)
    if index < 0:
        marker = '"shape":'
        index = header.find(marker)
    if index < 0:
        raise ValueError("Missing .npy shape")
    start = header.find("(", index)
    end = header.find(")", start)
    if start < 0 or end < 0:
        raise ValueError("Invalid .npy shape")
    parts = [part.strip() for part in header[start + 1 : end].split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Expected 2D .npy mask")
    return int(parts[0]), int(parts[1])


def _check_cancelled(context: JobContext) -> bool:
    return context.is_cancelled()


def _progress_for_cache(job_id: int) -> runtime_env.ProgressCallback:
    def progress(stage: str, value: float | None, extra: dict[str, object]) -> None:
        message: JsonObject = {"id": job_id, "event": "progress", "stage": stage, "progress": value}
        message.update(extra)
        emit(message)

    return progress


def handle_prepare_selection(job_id: int, params: Mapping[str, Any], context: JobContext) -> None:
    try:
        width = _positive_int(params.get("canvas_width"), "canvas width")
        height = _positive_int(params.get("canvas_height"), "canvas height")
        image_path = _required_path(params, "image_path")
        alpha_mask_path = _required_path(params, "alpha_mask_path")
        region_mask_path = _required_path(params, "region_mask_path")
    except ValueError as exc:
        emit_error(job_id, "ProtocolError", str(exc))
        return
    region_bbox = params.get("region_bbox")
    if not isinstance(region_bbox, list):
        emit_error(job_id, "EmptyRegion", "lasso region is empty")
        return

    model_id_value = params.get("model_id", default_model_id())
    model_id = (
        model_id_value if isinstance(model_id_value, str) and model_id_value else default_model_id()
    )
    try:
        model_spec = get_model_spec(model_id)
    except ModelRegistryError:
        emit_error(job_id, "UnknownModel", "unknown model")
        return

    runtime_cache_dir = params.get("runtime_cache_dir")
    if not isinstance(runtime_cache_dir, str) or not runtime_cache_dir:
        emit_error(job_id, "RuntimeCacheDirMissing", "runtime cache is not writable")
        return

    model_cache_dir = params.get("model_cache_dir")
    if not isinstance(model_cache_dir, str) or not model_cache_dir:
        emit_error(job_id, "ModelCacheDirMissing", "model cache is not writable")
        return

    log_event(
        "selection.start",
        component="worker",
        **{
            "job.id": job_id,
            "job.type": "lasso_selection",
            "image.width": width,
            "image.height": height,
        },
    )

    try:
        runtime_ready = runtime_env.ensure_runtime(
            Path(runtime_cache_dir), _progress_for_cache(job_id), context
        )
    except runtime_env.RuntimeEnvError as exc:
        emit_error(job_id, error_type_for_runtime_error(exc), str(exc))
        return

    if _check_cancelled(context):
        return

    try:
        selection_input = region_selection.load_region_selection_input(
            image_path,
            alpha_mask_path,
            region_mask_path,
            width,
            height,
            region_bbox,
            read_npy_uint8_2d,
        )
    except region_selection.RegionSelectionError as exc:
        emit_error(job_id, "EmptyRegion", str(exc))
        return

    if _check_cancelled(context):
        return

    emit_progress(job_id, "resolving_model")
    model_cache = ModelCache(Path(model_cache_dir))
    try:
        model_status = model_cache.ensure_downloaded(
            model_spec, _progress_for_cache(job_id), context
        )
    except ModelDownloadCancelled:
        return
    except ModelCacheError as exc:
        emit_error(job_id, error_type_for_model_cache_error(exc), str(exc))
        return

    if model_status.snapshot_dir is None:
        emit_error(job_id, "ModelVerificationError", "model verification failed")
        return
    if _check_cancelled(context):
        return

    emit_progress(job_id, "loading_model")
    emit_progress(job_id, "running_alpha_model")
    try:
        inference_result = birefnet_inference.predict_alpha(
            model_spec, model_status.snapshot_dir, runtime_ready, selection_input.crop_image_array
        )
    except birefnet_inference.BiRefNetInferenceError as exc:
        emit_error(job_id, error_type_for_inference_error(exc), str(exc))
        return

    if _check_cancelled(context):
        return

    emit_progress(job_id, "building_alpha_mask")
    try:
        build_result = alpha_selection_builder.build_selection_mask_from_crop_alpha(
            inference_result.crop_alpha,
            selection_input,
            score=1.0,
        )
    except alpha_selection_builder.EmptySelectionError as exc:
        emit_error(job_id, "EmptySelection", str(exc))
        return

    if _check_cancelled(context):
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-result-"))
    context.temp_paths.append(temp_dir)
    selection_mask_path = temp_dir / "selection_mask.npy"
    write_npy(selection_mask_path, "|u1", (height, width), build_result.selection_mask)
    _write_debug_exports(selection_input, inference_result, build_result)

    result: JsonObject = {
        "selection_mask_path": str(selection_mask_path),
        "metadata": build_result.metadata,
        "model": model_result_payload(model_spec, model_status),
        "runtime": birefnet_inference.runtime_result_payload(inference_result),
    }
    emit({"id": job_id, "result": result})


def model_result_payload(spec: ModelSpec, status: ModelCacheStatus) -> dict[str, object]:
    return {
        "id": spec.id,
        "title": spec.title,
        "cache_state": status.state,
        "checkpoint_bytes": status.checkpoint_bytes,
    }


def _write_debug_exports(
    selection_input: region_selection.RegionSelectionInput,
    inference_result: birefnet_inference.BiRefNetResult,
    build_result: alpha_selection_builder.AlphaSelectionBuildResult,
) -> None:
    if not os.environ.get("SMART_SELECT_KEEP_DEBUG_FILES"):
        return

    try:
        image_module = __import__("PIL.Image", fromlist=["Image"])
        debug_dir = Path(tempfile.mkdtemp(prefix="krita-smart-select-debug-")) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop_width = selection_input.crop_bbox[2]
        crop_height = selection_input.crop_bbox[3]
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
            image_module,
            debug_dir / "birefnet_alpha_raw.png",
            inference_result.crop_alpha,
            crop_width,
            crop_height,
        )
        _save_debug_mask(
            image_module,
            debug_dir / "final_alpha.png",
            build_result.selection_mask,
            selection_input.canvas_width,
            selection_input.canvas_height,
        )
        metadata = {
            "model": {
                "id": "birefnet_hr_matting",
                "repo_id": "ZhengPeng7/BiRefNet_HR-matting",
                "revision": "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
            },
            "crop": {
                "bbox": list(selection_input.crop_bbox),
                "width": crop_width,
                "height": crop_height,
            },
            "alpha": {
                "raw_nonzero_pixels": build_result.raw_nonzero_pixels,
                "final_nonzero_pixels": build_result.final_nonzero_pixels,
                "soft_pixels": build_result.soft_pixels,
                "max_alpha": build_result.max_alpha,
                "mean_nonzero_alpha": build_result.mean_nonzero_alpha,
            },
        }
        (debug_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
            **{"error.type": exc.__class__.__name__, "error.message": str(exc)},
        )


def _save_debug_mask(image_module: Any, path: Path, data: bytes, width: int, height: int) -> None:
    image = image_module.frombytes("L", (width, height), data)
    image.save(path)


def error_type_for_runtime_error(exc: runtime_env.RuntimeEnvError) -> str:
    if isinstance(exc, runtime_env.RuntimeCacheDirMissing):
        return "RuntimeCacheDirMissing"
    if isinstance(exc, runtime_env.RuntimeDependencyError):
        return "RuntimeDependencyError"
    if isinstance(exc, runtime_env.RuntimeValidationError):
        return "RuntimeValidationError"
    return "RuntimeBootstrapError"


def error_type_for_model_cache_error(exc: ModelCacheError) -> str:
    if isinstance(exc, ModelCacheDirMissing):
        return "ModelCacheDirMissing"
    if isinstance(exc, ModelDownloadError):
        return "ModelDownloadError"
    if isinstance(exc, ModelVerificationError):
        return "ModelVerificationError"
    return "ModelDownloadError"


def error_type_for_inference_error(exc: birefnet_inference.BiRefNetInferenceError) -> str:
    if isinstance(exc, birefnet_inference.InferenceOutOfMemory):
        return "InferenceOutOfMemory"
    return "InferenceFailed"


def _required_path(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {key}")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid {name}")
    return value


def handle_diagnostics(job_id: int, params: JsonObject) -> None:
    model_cache_dir_value = params.get("model_cache_dir")
    runtime_cache_dir_value = params.get("runtime_cache_dir")
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
    emit(
        {
            "id": job_id,
            "result": collect_runtime_diagnostics(model_cache_dir, runtime_cache_dir),
        }
    )


def dispatch_job(context: JobContext, params: Mapping[str, Any]) -> None:
    method = params.get("_method")
    if method == "prepare_selection":
        handle_prepare_selection(context.job_id, params, context)
        return
    emit_error(context.job_id, "ProtocolError", "unknown method")


def ensure_runner() -> JobRunner:
    global _job_runner
    if _job_runner is None:
        _job_runner = JobRunner(dispatch_job, emit_error)
    return _job_runner


def handle_message(message: JsonObject) -> None:
    job_id = int(message.get("id", 0))
    method = message.get("method")
    params_value = message.get("params", {})
    params = params_value if isinstance(params_value, dict) else {}
    if method == "prepare_selection":
        ensure_runner().start(job_id, {"_method": "prepare_selection", **params})
        return
    if method == "cancel":
        target = params.get("job_id")
        if isinstance(target, int):
            ensure_runner().cancel(target)
        emit({"id": job_id, "result": {"cancelled": True}})
        return
    if method == "diagnostics":
        handle_diagnostics(job_id, params)
        return
    emit_error(job_id, "ProtocolError", "unknown method")


def main() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
            handle_message(message)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            emit_error(0, "ProtocolError", "invalid worker message")
    if _job_runner is not None:
        _job_runner.wait_for_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
