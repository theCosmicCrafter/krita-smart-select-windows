from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_registry import ModelSpec
from observability import log_event
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


@dataclass(frozen=True)
class PromptedSelectionResult:
    masks: Sequence[Any]
    scores: Sequence[float]
    device: str
    torch_version: str | None
    cuda_available: bool


_GENERATOR_CACHE: dict[tuple[str, str, str, str], Any] = {}
_PREDICTOR_CACHE: dict[tuple[str, str, str, str], Any] = {}


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
            log_event(
                "inference.failed",
                severity="error",
                component="sam2",
                **{
                    "error.type": "InferenceOutOfMemory",
                    "model.id": spec.id,
                    "device.type": device,
                },
            )
            raise InferenceOutOfMemory("SAM2 ran out of GPU memory") from exc
        log_event(
            "inference.failed",
            severity="error",
            component="sam2",
            **{
                "error.type": exc.__class__.__name__,
                "model.id": spec.id,
                "device.type": device,
            },
        )
        raise InferenceFailed("SAM2 inference failed") from exc

    duration_ms = int((time.monotonic() - started) * 1000)
    log_event(
        "sam.generate.result",
        component="sam2",
        **{
            "model.id": spec.id,
            "device.type": device,
            "masks.count": len(masks),
            "duration_ms": duration_ms,
        },
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


def predict_selection(
    spec: ModelSpec,
    checkpoint_path: Path,
    image_array: Any,
    box_xyxy: Sequence[float],
    point_coords: Sequence[Sequence[float]],
    point_labels: Sequence[int],
) -> PromptedSelectionResult:
    if spec.config_name is None:
        raise InferenceFailed("SAM2 config is missing")

    torch = importlib.import_module("torch")
    numpy_module = importlib.import_module("numpy")
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"

    started = time.monotonic()
    try:
        predictor = _predictor_for(spec, checkpoint_path, device)
        with torch.inference_mode():
            predictor.set_image(image_array)
            masks, scores, _logits = predictor.predict(
                point_coords=numpy_module.array(point_coords, dtype=numpy_module.float32),
                point_labels=numpy_module.array(point_labels, dtype=numpy_module.int32),
                box=numpy_module.array(box_xyxy, dtype=numpy_module.float32),
                multimask_output=True,
                return_logits=False,
            )
    except Exception as exc:
        if _is_cuda_oom(exc):
            log_event(
                "inference.failed",
                severity="error",
                component="sam2",
                **{
                    "error.type": "InferenceOutOfMemory",
                    "model.id": spec.id,
                    "device.type": device,
                },
            )
            raise InferenceOutOfMemory("SAM2 ran out of GPU memory") from exc
        log_event(
            "inference.failed",
            severity="error",
            component="sam2",
            **{
                "error.type": exc.__class__.__name__,
                "model.id": spec.id,
                "device.type": device,
            },
        )
        raise InferenceFailed("SAM2 inference failed") from exc

    mask_list = _sequence_from_array(masks)
    score_list = [float(value) for value in _sequence_from_array(scores)]
    duration_ms = int((time.monotonic() - started) * 1000)
    log_event(
        "sam.predict.result",
        component="sam2",
        **{
            "model.id": spec.id,
            "device.type": device,
            "masks.count": len(mask_list),
            "duration_ms": duration_ms,
        },
    )
    return PromptedSelectionResult(
        masks=mask_list,
        scores=score_list,
        device=device,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        cuda_available=cuda_available,
    )


def runtime_prompted_result_payload(result: PromptedSelectionResult) -> dict[str, object]:
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

    log_event(
        "model.load.start",
        component="sam2",
        **{"model.id": spec.id, "device.type": device},
    )
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


def _predictor_for(spec: ModelSpec, checkpoint_path: Path, device: str) -> Any:
    if spec.config_name is None:
        raise InferenceDeviceError("SAM2 config is missing")

    key = (spec.id, str(checkpoint_path), spec.config_name, device)
    cached = _PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached

    build_sam = importlib.import_module("sam2.build_sam")
    predictor_module = importlib.import_module("sam2.sam2_image_predictor")
    build_sam2 = build_sam.build_sam2
    predictor_class = predictor_module.SAM2ImagePredictor

    log_event(
        "model.load.start",
        component="sam2",
        **{"model.id": spec.id, "device.type": device},
    )
    try:
        model = build_sam2(spec.config_name, str(checkpoint_path), device=device)
        predictor = predictor_class(
            model, mask_threshold=0.0, max_hole_area=64, max_sprinkle_area=64
        )
    except TypeError as exc:
        raise RuntimeError("SAM2 predictor parameter mismatch") from exc
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise InferenceOutOfMemory("SAM2 ran out of GPU memory") from exc
        raise InferenceDeviceError("SAM2 model failed to load") from exc

    _PREDICTOR_CACHE[key] = predictor
    return predictor


def _sequence_from_array(value: Any) -> list[Any]:
    try:
        return [value[index] for index in range(len(value))]
    except TypeError:
        return [value]


def _is_cuda_oom(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "cuda" in text and ("out of memory" in text or "oom" in text)
