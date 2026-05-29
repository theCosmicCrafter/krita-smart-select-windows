from __future__ import annotations

import importlib
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from observability import log_event
from refiner_cache import RefinerCacheStatus
from refiner_registry import RefinerSpec
from runtime_env import MamSourceReady, RuntimeReady
from trimap_builder import TrimapBuildResult

MAM_IMAGE_SIZE = 1024
OS8_WIDTH = 10
OS4_WIDTH = 20
OS1_WIDTH = 10
MAX_MAM_ADDED_VS_PRIOR_FRACTION = 0.03
MAX_MAM_ADDED_VS_PRIOR_PIXELS = 512
MAX_MAM_REMOVED_VS_PRIOR_FRACTION = 0.08
MAX_MAM_REMOVED_VS_PRIOR_PIXELS = 1024
MIN_MAM_AREA_VS_PRIOR_RATIO = 0.70
MAX_MAM_AREA_VS_PRIOR_RATIO = 1.12

_MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}


class MamRefinerError(RuntimeError):
    pass


class MattingFailed(MamRefinerError):
    pass


class MattingOutOfMemory(MamRefinerError):
    pass


@dataclass(frozen=True)
class MamRefinementInput:
    crop_rgb: object
    crop_alpha_mask: bytes
    crop_region_mask: bytes
    crop_sam_prior_mask: bytes
    trimap: TrimapBuildResult
    crop_bbox: tuple[int, int, int, int]
    prompt_box_xyxy: tuple[float, float, float, float]
    canvas_width: int
    canvas_height: int
    prior_fallback_used: bool
    prior_region_coverage: float


@dataclass(frozen=True)
class MamRefinementResult:
    crop_alpha: bytes
    device: str
    torch_version: str | None
    cuda_available: bool
    duration_ms: int
    fallback_used: bool = False
    rejected: bool = False
    rejection_reason: str | None = None


def refine_with_mam(
    spec: RefinerSpec,
    cache_status: RefinerCacheStatus,
    runtime_ready: RuntimeReady,
    source_ready: MamSourceReady,
    refinement_input: MamRefinementInput,
) -> MamRefinementResult:
    if cache_status.sam_checkpoint_path is None or cache_status.mam_checkpoint_path is None:
        raise MattingFailed("edge refinement failed")

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    started = time.monotonic()

    try:
        model = _model_for(spec, cache_status, source_ready, device)
        sample = _sample_from_input(refinement_input, source_ready.source_root, device)
        with torch.inference_mode():
            _feas, pred, _post_mask = model.forward_inference(sample)
            alpha = _compose_alpha(pred, refinement_input, device)
        gate = _refinement_gate(alpha, refinement_input.crop_sam_prior_mask)
        log_event(
            "refinement.gate.result",
            component="mam",
            **{
                "refiner.id": spec.id,
                "refinement.accepted": gate.accepted,
                "refinement.rejection_reason": gate.reason,
                "refinement.added_vs_prior_pixels": gate.added_vs_prior_pixels,
                "refinement.removed_vs_prior_pixels": gate.removed_vs_prior_pixels,
                "refinement.area_ratio_vs_prior": round(gate.area_ratio_vs_prior, 4),
            },
        )
        if not gate.accepted:
            log_event(
                "refinement.rejected",
                severity="warning",
                component="mam",
                **{
                    "refiner.id": spec.id,
                    "error.reason": gate.reason,
                    "refinement.added_vs_prior_pixels": gate.added_vs_prior_pixels,
                    "refinement.removed_vs_prior_pixels": gate.removed_vs_prior_pixels,
                    "refinement.area_ratio_vs_prior": round(gate.area_ratio_vs_prior, 4),
                },
            )
            alpha = refinement_input.crop_sam_prior_mask
            rejected = True
            rejection_reason = gate.reason
        else:
            rejected = False
            rejection_reason = None
    except Exception as exc:
        if _is_cuda_oom(exc):
            log_event(
                "refinement.failed",
                severity="error",
                component="mam",
                **{
                    "error.type": "MattingOutOfMemory",
                    "refiner.id": spec.id,
                    "device.type": device,
                },
            )
            raise MattingOutOfMemory("edge refinement ran out of GPU memory") from exc
        log_event(
            "refinement.failed",
            severity="error",
            component="mam",
            **{
                "error.type": exc.__class__.__name__,
                "error.reason": str(exc),
                "refiner.id": spec.id,
                "device.type": device,
            },
        )
        raise MattingFailed("edge refinement failed") from exc

    duration_ms = int((time.monotonic() - started) * 1000)
    log_event(
        "mam.refine.result",
        component="mam",
        **{
            "refiner.id": spec.id,
            "device.type": device,
            "duration_ms": duration_ms,
        },
    )
    return MamRefinementResult(
        crop_alpha=alpha,
        device=device,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        cuda_available=cuda_available,
        duration_ms=duration_ms,
        fallback_used=rejected,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def runtime_result_payload(result: MamRefinementResult, spec: RefinerSpec) -> dict[str, object]:
    return {
        "id": spec.id,
        "title": spec.title,
        "cache_state": (
            "rejected_mam"
            if result.rejected
            else "fallback_sam_prior"
            if result.fallback_used
            else "verified"
        ),
        "device": result.device,
        "torch_version": result.torch_version,
        "cuda_available": result.cuda_available,
        "sam_checkpoint_bytes": spec.sam_expected_bytes,
        "mam_checkpoint_bytes": spec.mam_expected_bytes,
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason,
    }


def fallback_from_sam_prior(
    spec: RefinerSpec, prior_mask: bytes, reason: str
) -> MamRefinementResult:
    torch_version: str | None = None
    cuda_available = False
    device = "cpu"
    try:
        torch = importlib.import_module("torch")
        torch_version = str(getattr(torch, "__version__", "")) or None
        cuda_available = bool(torch.cuda.is_available())
        device = "cuda" if cuda_available else "cpu"
    except Exception:
        pass
    log_event(
        "refinement.failed",
        severity="warning",
        component="mam",
        **{
            "refiner.id": spec.id,
            "error.reason": reason,
            "model.cache.state": "fallback_sam_prior",
        },
    )
    return MamRefinementResult(
        crop_alpha=prior_mask,
        device=device,
        torch_version=torch_version,
        cuda_available=cuda_available,
        duration_ms=0,
        fallback_used=True,
    )


def _model_for(
    spec: RefinerSpec,
    cache_status: RefinerCacheStatus,
    source_ready: MamSourceReady,
    device: str,
) -> Any:
    if cache_status.sam_checkpoint_path is None or cache_status.mam_checkpoint_path is None:
        raise MattingFailed("edge refinement failed")
    key = (
        spec.id,
        str(cache_status.mam_checkpoint_path),
        str(cache_status.sam_checkpoint_path),
        device,
    )
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    _prepare_imports(source_ready.source_root)
    _ensure_sam_checkpoint_link(source_ready.source_root, spec, cache_status.sam_checkpoint_path)
    torch = importlib.import_module("torch")
    networks = importlib.import_module("networks")
    log_event(
        "mam.load.start",
        component="mam",
        **{"refiner.id": spec.id, "device.type": device},
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(source_ready.source_root)
        model = networks.get_generator_m2m(seg=spec.sam_model_type, m2m="sam_decoder_deep")
    finally:
        os.chdir(old_cwd)

    model = model.to(device)
    checkpoint = torch.load(str(cache_status.mam_checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.m2m.load_state_dict(_remove_prefix_state_dict(state_dict), strict=True)
    model.eval()
    _MODEL_CACHE[key] = model
    log_event(
        "mam.load.result",
        component="mam",
        **{"refiner.id": spec.id, "device.type": device},
    )
    return model


def _prepare_imports(source_root: Path) -> None:
    _install_fake_utils_module()
    segment_anything_path = source_root / "segment-anything"
    for path in (source_root, segment_anything_path):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _install_fake_utils_module() -> None:
    config = types.SimpleNamespace()
    config.model = types.SimpleNamespace()
    config.model.arch = types.SimpleNamespace(seg="sam_vit_h", m2m="sam_decoder_deep")
    module = types.ModuleType("utils")
    module_any = cast(Any, module)
    module_any.CONFIG = config
    module_any.__smart_select_fake__ = True
    sys.modules["utils"] = module


def _ensure_sam_checkpoint_link(
    source_root: Path, spec: RefinerSpec, sam_checkpoint_path: Path
) -> None:
    checkpoint_dir = source_root / "segment-anything" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_path = checkpoint_dir / spec.sam_checkpoint_filename
    if expected_path.exists():
        return
    try:
        expected_path.symlink_to(sam_checkpoint_path)
    except OSError:
        import shutil

        shutil.copy2(sam_checkpoint_path, expected_path)


def _sample_from_input(
    refinement_input: MamRefinementInput,
    source_root: Path,
    device: str,
) -> dict[str, Any]:
    _prepare_imports(source_root)
    numpy_module = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    transforms = importlib.import_module("segment_anything.utils.transforms")
    transform = transforms.ResizeLongestSide(MAM_IMAGE_SIZE)

    image_array = numpy_module.asarray(refinement_input.crop_rgb)
    original_size = (int(image_array.shape[0]), int(image_array.shape[1]))
    transformed = transform.apply_image(image_array)
    image = torch.as_tensor(transformed, device=device)
    image = image.permute(2, 0, 1).contiguous()
    box = numpy_module.array([refinement_input.prompt_box_xyxy], dtype=numpy_module.float32)
    transformed_box = transform.apply_boxes(box, original_size)
    bbox = torch.as_tensor(transformed_box, dtype=torch.float, device=device)

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)
    image = (image - pixel_mean) / pixel_std

    height, width = image.shape[-2:]
    pad_height = MAM_IMAGE_SIZE - height
    pad_width = MAM_IMAGE_SIZE - width
    image = functional.pad(image, (0, pad_width, 0, pad_height))
    return {
        "image": image[None, ...],
        "bbox": bbox[None, ...],
        "ori_shape": original_size,
        "pad_shape": (height, width),
    }


def _compose_alpha(
    pred: dict[str, Any],
    refinement_input: MamRefinementInput,
    device: str,
) -> bytes:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    crop_width = refinement_input.crop_bbox[2]
    crop_height = refinement_input.crop_bbox[3]
    shape = (crop_height, crop_width)
    alpha_os8 = functional.interpolate(
        pred["alpha_os8"], shape, mode="bilinear", align_corners=False
    )
    alpha_os4 = functional.interpolate(
        pred["alpha_os4"], shape, mode="bilinear", align_corners=False
    )
    alpha_os1 = functional.interpolate(
        pred["alpha_os1"], shape, mode="bilinear", align_corners=False
    )
    core = _mask_tensor(refinement_input.trimap.core_mask, crop_width, crop_height, device)
    unknown = _mask_tensor(refinement_input.trimap.unknown_mask, crop_width, crop_height, device)
    support = _mask_tensor(refinement_input.trimap.support_mask, crop_width, crop_height, device)
    log_event(
        "mam.guidance.selected",
        component="mam",
        **{
            "refiner.guidance.source": "sam2_trimap",
            "selection.prior.fallback": refinement_input.prior_fallback_used,
            "selection.prior.region_coverage": round(refinement_input.prior_region_coverage, 4),
            "trimap.core.area": refinement_input.trimap.core_area,
            "trimap.unknown.area": refinement_input.trimap.unknown_area,
            "trimap.support.area": refinement_input.trimap.support_area,
        },
    )
    alpha = torch.zeros_like(core)
    alpha[core > 0] = 1.0

    weight_os8 = unknown
    alpha[weight_os8 > 0] = alpha_os8[weight_os8 > 0]
    weight_os4 = _unknown_from_pred_oneside(alpha, OS4_WIDTH) * unknown
    alpha[weight_os4 > 0] = alpha_os4[weight_os4 > 0]
    weight_os1 = _unknown_from_pred_oneside(alpha, OS1_WIDTH) * unknown
    alpha[weight_os1 > 0] = alpha_os1[weight_os1 > 0]

    region = _mask_tensor(refinement_input.crop_region_mask, crop_width, crop_height, device)
    document_alpha = _mask_tensor(refinement_input.crop_alpha_mask, crop_width, crop_height, device)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    alpha = torch.where(core > 0, torch.ones_like(alpha), alpha)
    alpha = torch.where(support > 0, alpha, torch.zeros_like(alpha))
    alpha = torch.where(region > 0, alpha, torch.zeros_like(alpha))
    alpha = torch.where(document_alpha > 0, alpha, torch.zeros_like(alpha))
    output = torch.clamp(alpha[0, 0] * 255.0, 0, 255).to(torch.uint8).cpu().numpy()
    return bytes(output.tobytes())


def _guidance_source(refinement_input: MamRefinementInput) -> str:
    return "sam2_prior"


def _mask_tensor(mask: bytes, width: int, height: int, device: str) -> Any:
    torch = importlib.import_module("torch")
    tensor = torch.tensor(list(mask), dtype=torch.float32, device=device).reshape(
        1, 1, height, width
    )
    return tensor / 255.0


def _unknown_from_pred_oneside(pred: Any, width: int) -> Any:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    safe_width = max(1, width // 2)
    uncertain = (pred >= (1.0 / 255.0)).to(torch.float32)
    dilated = functional.max_pool2d(
        uncertain, kernel_size=safe_width, stride=1, padding=safe_width // 2
    )
    dilated = dilated[..., : pred.shape[-2], : pred.shape[-1]]
    return torch.where(pred > (1.0 - 1.0 / 255.0), torch.zeros_like(dilated), dilated)


def _remove_prefix_state_dict(state_dict: object, prefix: str = "module.") -> dict[str, Any]:
    if not isinstance(state_dict, dict):
        raise MattingFailed("edge refinement failed")
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        key_text = str(key)
        result[key_text[len(prefix) :] if key_text.startswith(prefix) else key_text] = value
    return result


def _is_cuda_oom(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "cuda" in text and ("out of memory" in text or "oom" in text)


@dataclass(frozen=True)
class _RefinementGateResult:
    accepted: bool
    reason: str | None
    added_vs_prior_pixels: int
    removed_vs_prior_pixels: int
    area_ratio_vs_prior: float


def _refinement_gate(candidate_alpha: bytes, prior_mask: bytes) -> _RefinementGateResult:
    if len(candidate_alpha) != len(prior_mask):
        return _RefinementGateResult(
            accepted=False,
            reason="size_mismatch",
            added_vs_prior_pixels=0,
            removed_vs_prior_pixels=0,
            area_ratio_vs_prior=0.0,
        )
    prior_area = 0
    candidate_area = 0
    added = 0
    removed = 0
    for candidate_value, prior_value in zip(candidate_alpha, prior_mask, strict=True):
        prior_on = prior_value > 0
        candidate_on = candidate_value > 0
        if prior_on:
            prior_area += 1
        if candidate_on:
            candidate_area += 1
        if candidate_on and not prior_on:
            added += 1
        if prior_on and not candidate_on:
            removed += 1

    if prior_area <= 0:
        return _RefinementGateResult(
            accepted=candidate_area > 0,
            reason=None if candidate_area > 0 else "empty_candidate",
            added_vs_prior_pixels=added,
            removed_vs_prior_pixels=removed,
            area_ratio_vs_prior=0.0,
        )

    area_ratio = candidate_area / prior_area
    max_added = max(
        MAX_MAM_ADDED_VS_PRIOR_PIXELS,
        int(prior_area * MAX_MAM_ADDED_VS_PRIOR_FRACTION),
    )
    max_removed = max(
        MAX_MAM_REMOVED_VS_PRIOR_PIXELS,
        int(prior_area * MAX_MAM_REMOVED_VS_PRIOR_FRACTION),
    )
    reason: str | None = None
    if added > max_added:
        reason = "added_too_much"
    elif removed > max_removed:
        reason = "removed_too_much"
    elif area_ratio < MIN_MAM_AREA_VS_PRIOR_RATIO:
        reason = "area_too_small"
    elif area_ratio > MAX_MAM_AREA_VS_PRIOR_RATIO:
        reason = "area_too_large"

    return _RefinementGateResult(
        accepted=reason is None,
        reason=reason,
        added_vs_prior_pixels=added,
        removed_vs_prior_pixels=removed,
        area_ratio_vs_prior=area_ratio,
    )
