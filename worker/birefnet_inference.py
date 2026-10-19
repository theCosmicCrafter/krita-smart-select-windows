from __future__ import annotations

import gc
import importlib
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_model_registry import ModelSpec
from observability import log_event
from runtime_env import RuntimeReady


class BiRefNetInferenceError(RuntimeError):
    pass


class InferenceOutOfMemory(BiRefNetInferenceError):
    pass


class InferenceFailed(BiRefNetInferenceError):
    pass


class InferenceUnsupportedGpu(BiRefNetInferenceError):
    pass


@dataclass(frozen=True)
class BiRefNetResult:
    crop_alpha: bytes
    width: int
    height: int
    device: str
    torch_version: str | None
    cuda_available: bool
    duration_ms: int


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

_SM_ARCH = re.compile(r"sm_(\d+)[a-z]?")


@dataclass(frozen=True)
class CudaDevice:
    name: str
    capability: tuple[int, int]
    arch_list: tuple[str, ...]

    @property
    def compute_capability(self) -> str:
        return f"{self.capability[0]}.{self.capability[1]}"

    @property
    def supported(self) -> bool:
        """Whether the installed torch ships kernels for this GPU.

        Same rule as torch's own warning: NVIDIA architectures are only compatible
        within a major version, so e.g. an sm_120 GPU needs an sm_12x build. A torch
        build without any `sm_` entry (PTX only) cannot be judged and counts as supported.
        """
        supported_majors = {
            int(match.group(1)) // 10
            for arch in self.arch_list
            if (match := _SM_ARCH.fullmatch(arch))
        }
        return not supported_majors or self.capability[0] in supported_majors


def cuda_device(torch: Any) -> CudaDevice | None:
    """Describe CUDA device 0, or None when torch cannot be asked."""
    try:
        major, minor = (int(value) for value in torch.cuda.get_device_capability(0))
        arch_list = tuple(str(arch) for arch in torch.cuda.get_arch_list())
        name = str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return CudaDevice(name, (major, minor), arch_list)


def release_runtime_memory() -> None:
    model_count = len(_MODEL_CACHE)
    torch: Any | None = None
    cuda_available = False
    allocated_before: int | None = None
    reserved_before: int | None = None
    allocated_after: int | None = None
    reserved_after: int | None = None

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            allocated_before = int(torch.cuda.memory_allocated())
            reserved_before = int(torch.cuda.memory_reserved())
    except Exception:
        torch = None

    models = list(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()
    for cached_model in models:
        with suppress(Exception):
            cached_model.to("cpu")
    del models
    if "cached_model" in locals():
        del cached_model
    gc.collect()

    if torch is not None and cuda_available:
        with suppress(Exception):
            torch.cuda.empty_cache()
        with suppress(Exception):
            torch.cuda.ipc_collect()
        try:
            allocated_after = int(torch.cuda.memory_allocated())
            reserved_after = int(torch.cuda.memory_reserved())
        except Exception:
            pass

    log_event(
        "alpha_model.memory.released",
        component="birefnet",
        **{
            "model.cache.entries": model_count,
            "cuda.available": cuda_available,
            "cuda.memory.allocated_before": allocated_before,
            "cuda.memory.reserved_before": reserved_before,
            "cuda.memory.allocated_after": allocated_after,
            "cuda.memory.reserved_after": reserved_after,
        },
    )


def predict_alpha(
    spec: ModelSpec,
    snapshot_dir: Path,
    _runtime_ready: RuntimeReady,
    crop_image_array: object,
) -> BiRefNetResult:
    torch = importlib.import_module("torch")
    image_module = importlib.import_module("PIL.Image")
    numpy_module = importlib.import_module("numpy")
    transforms = importlib.import_module("torchvision.transforms")

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        _require_supported_gpu(torch)
    crop_image = image_module.fromarray(crop_image_array).convert("RGB")
    width, height = crop_image.size

    try:
        model = _load_model(spec, snapshot_dir, device)
        image_tensor = _preprocess_image(crop_image, spec.input_resolution, transforms)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        if device == "cuda":
            image_tensor = image_tensor.half()

        log_event(
            "alpha_model.predict.start",
            component="birefnet",
            **{
                "model.id": spec.id,
                "device.type": device,
                "image.width": width,
                "image.height": height,
            },
        )
        started = time.monotonic()
        with torch.inference_mode():
            prediction = model(image_tensor)[-1].sigmoid().detach().cpu()
        duration_ms = round((time.monotonic() - started) * 1000)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            _log_predict_failed(spec, device, exc)
            raise InferenceOutOfMemory("alpha extraction ran out of GPU memory") from exc
        if "no kernel image" in message:
            # CUDA could not run the model on this GPU even though the arch check passed.
            _log_predict_failed(spec, device, exc)
            raise InferenceUnsupportedGpu(
                "the installed PyTorch has no CUDA kernels for this GPU"
                f" (torch {torch.__version__})"
            ) from exc
        _log_predict_failed(spec, device, exc)
        raise InferenceFailed(f"alpha extraction failed ({device}, {type(exc).__name__})") from exc
    except Exception as exc:
        _log_predict_failed(spec, device, exc)
        raise InferenceFailed(f"alpha extraction failed ({device}, {type(exc).__name__})") from exc

    pred = prediction[0].squeeze().float()
    pred_image = transforms.ToPILImage()(pred)
    resampling = getattr(image_module, "Resampling", image_module).BILINEAR
    pred_image = pred_image.resize((width, height), resampling)
    alpha_array = numpy_module.asarray(pred_image, dtype=numpy_module.uint8)
    alpha_bytes = alpha_array.reshape(width * height).tobytes()
    log_event(
        "alpha_model.predict.result",
        component="birefnet",
        **{
            "model.id": spec.id,
            "device.type": device,
            "duration_ms": duration_ms,
            "image.width": width,
            "image.height": height,
        },
    )
    return BiRefNetResult(
        crop_alpha=alpha_bytes,
        width=width,
        height=height,
        device=device,
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        duration_ms=duration_ms,
    )


def _require_supported_gpu(torch: Any) -> None:
    device = cuda_device(torch)
    if device is None:
        return
    fields: dict[str, object] = {
        "device.type": "cuda",
        "device.name": device.name,
        "device.compute_capability": device.compute_capability,
        "torch.version": str(torch.__version__),
        "torch.cuda.arch_list": list(device.arch_list),
    }
    if device.supported:
        log_event("alpha_model.cuda.checked", component="birefnet", **fields)
        return
    log_event(
        "alpha_model.cuda.incompatible",
        severity="error",
        component="birefnet",
        **fields,
        **{"error.type": "InferenceUnsupportedGpu", "error.reason": "cuda_arch_not_supported"},
    )
    raise InferenceUnsupportedGpu(
        f"{device.name} (compute capability {device.compute_capability}) is not supported by"
        f" the installed PyTorch {torch.__version__} (built for {' '.join(device.arch_list)});"
        " the runtime needs a PyTorch build with CUDA 12.8 or newer"
    )


def _log_predict_failed(spec: ModelSpec, device: str, exc: Exception) -> None:
    log_event(
        "alpha_model.predict.failed",
        severity="error",
        component="birefnet",
        **{
            "model.id": spec.id,
            "device.type": device,
            "error.type": type(exc).__name__,
            "error.message": str(exc)[:500],
        },
    )


def runtime_result_payload(result: BiRefNetResult) -> dict[str, object]:
    return {
        "version": runtime_ready_version(),
        "cache_state": "ready",
        "device": result.device,
        "torch_version": result.torch_version,
        "cuda_available": result.cuda_available,
    }


def runtime_ready_version() -> str:
    runtime_manifest = importlib.import_module("runtime_manifest")
    return str(runtime_manifest.RUNTIME_VERSION)


def _load_model(spec: ModelSpec, snapshot_dir: Path, device: str) -> Any:
    key = (spec.id, str(snapshot_dir), device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.set_float32_matmul_precision("high")
    log_event(
        "alpha_model.load.start",
        component="birefnet",
        **{"model.id": spec.id, "device.type": device},
    )
    model = transformers.AutoModelForImageSegmentation.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=spec.trust_remote_code,
    )
    model.to(device)
    model.eval()
    if device == "cuda":
        model.half()
    _MODEL_CACHE[key] = model
    log_event(
        "alpha_model.load.result",
        component="birefnet",
        **{"model.id": spec.id, "device.type": device},
    )
    return model


def _preprocess_image(image: Any, resolution: int, transforms: Any) -> Any:
    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)
