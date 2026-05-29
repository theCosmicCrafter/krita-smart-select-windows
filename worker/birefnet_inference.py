from __future__ import annotations

import importlib
import time
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
        if "out of memory" in str(exc).lower():
            raise InferenceOutOfMemory("alpha extraction ran out of GPU memory") from exc
        raise InferenceFailed("alpha extraction failed") from exc
    except Exception as exc:
        raise InferenceFailed("alpha extraction failed") from exc

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
