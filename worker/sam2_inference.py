from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_registry import ModelSpec
from runtime_manifest import RUNTIME_VERSION, SAM2_SOURCE_REF

GENERATOR_PARAMS: dict[str, object] = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.95,
    "stability_score_offset": 1.0,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 1,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.3413333333333333,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 64,
    "output_mode": "binary_mask",
    "use_m2m": True,
    "multimask_output": True,
}


class Sam2InferenceError(RuntimeError):
    pass


class InferenceDeviceError(Sam2InferenceError):
    pass


class InferenceOutOfMemory(Sam2InferenceError):
    pass


class InferenceFailed(Sam2InferenceError):
    pass


@dataclass(frozen=True)
class Sam2InferenceResult:
    masks: Sequence[Mapping[str, object]]
    device: str
    torch_version: str | None
    cuda_available: bool


_GENERATOR_CACHE: dict[tuple[str, str, str, str], Any] = {}


def generate_masks(
    spec: ModelSpec,
    checkpoint_path: Path,
    image_array: Any,
) -> Sam2InferenceResult:
    if spec.config_name is None:
        raise InferenceFailed("SAM2 config is missing")

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"

    started = time.monotonic()
    try:
        generator = _generator_for(spec, checkpoint_path, device)
        with torch.inference_mode():
            masks = generator.generate(image_array)
    except Exception as exc:
        if _is_cuda_oom(exc):
            _log_worker_event(
                "inference.failed",
                error_type="InferenceOutOfMemory",
                model_id=spec.id,
                device=device,
            )
            raise InferenceOutOfMemory("SAM2 ran out of GPU memory") from exc
        _log_worker_event(
            "inference.failed",
            error_type=exc.__class__.__name__,
            model_id=spec.id,
            device=device,
        )
        raise InferenceFailed("SAM2 inference failed") from exc

    duration_ms = int((time.monotonic() - started) * 1000)
    _log_worker_event(
        "sam.generate.result",
        model_id=spec.id,
        device=device,
        masks=len(masks),
        duration_ms=duration_ms,
    )
    return Sam2InferenceResult(
        masks=masks,
        device=device,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        cuda_available=cuda_available,
    )


def runtime_result_payload(result: Sam2InferenceResult) -> dict[str, object]:
    return {
        "version": RUNTIME_VERSION,
        "cache_state": "ready",
        "device": result.device,
        "torch_version": result.torch_version,
        "cuda_available": result.cuda_available,
        "sam2_source_ref": SAM2_SOURCE_REF,
    }


def _generator_for(spec: ModelSpec, checkpoint_path: Path, device: str) -> Any:
    if spec.config_name is None:
        raise InferenceDeviceError("SAM2 config is missing")

    key = (spec.id, str(checkpoint_path), spec.config_name, device)
    cached = _GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    build_sam = importlib.import_module("sam2.build_sam")
    generator_module = importlib.import_module("sam2.automatic_mask_generator")
    build_sam2 = build_sam.build_sam2
    generator_class = generator_module.SAM2AutomaticMaskGenerator

    _log_worker_event("model.load.start", model_id=spec.id, device=device)
    try:
        model = build_sam2(spec.config_name, str(checkpoint_path), device=device)
        generator = generator_class(model, **GENERATOR_PARAMS)
    except TypeError as exc:
        raise RuntimeError("SAM2 generator parameter mismatch") from exc
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise InferenceOutOfMemory("SAM2 ran out of GPU memory") from exc
        raise InferenceDeviceError("SAM2 model failed to load") from exc

    _GENERATOR_CACHE[key] = generator
    return generator


def _is_cuda_oom(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "cuda" in text and ("out of memory" in text or "oom" in text)


def _log_worker_event(event: str, **fields: object) -> None:
    parts = [f"[SmartSelectWorker] event={event}"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), file=sys.stderr, flush=True)
