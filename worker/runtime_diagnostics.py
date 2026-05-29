from __future__ import annotations

import importlib.util
import json
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from model_cache import ModelCache
from model_registry import all_model_specs
from refiner_cache import RefinerCache
from refiner_registry import all_refiner_specs
from runtime_env import runtime_status
from runtime_manifest import MAM_SOURCE_REF, RUNTIME_VERSION, SAM2_SOURCE_REF

JsonObject = dict[str, Any]


def _torch_diagnostics() -> JsonObject:
    if importlib.util.find_spec("torch") is None:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
        }

    try:
        torch = import_module("torch")
    except Exception:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = str(torch.cuda.get_device_name(0)) if cuda_available else None
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def _runtime_diagnostics(runtime_cache_dir: Path | None) -> JsonObject:
    if runtime_cache_dir is None:
        return {
            "cache_state": "missing",
            "runtime_version": RUNTIME_VERSION,
            "venv_exists": False,
            "python_executable": None,
            "sam2_source_ref": SAM2_SOURCE_REF,
            "sam2_importable": False,
            "torch_importable": False,
        }

    status = runtime_status(runtime_cache_dir)
    marker = _runtime_marker(status.runtime_json)
    return {
        "cache_state": status.state,
        "runtime_version": status.version or RUNTIME_VERSION,
        "venv_exists": status.venv_path.exists(),
        "python_executable": str(status.python_executable),
        "sam2_source_ref": status.sam2_source_ref or SAM2_SOURCE_REF,
        "cuda_home": marker.get("cuda_home"),
        "sam2_cuda_extension": marker.get("sam2_cuda_extension", False),
        "sam2_importable": False,
        "torch_importable": False,
    }


def _runtime_marker(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_runtime_diagnostics(
    model_cache_dir: Path | None,
    runtime_cache_dir: Path | None = None,
    refiner_cache_dir: Path | None = None,
) -> JsonObject:
    models: list[JsonObject] = []
    cache = ModelCache(model_cache_dir) if model_cache_dir is not None else None
    for model_spec in all_model_specs():
        if model_spec.id == "mock":
            continue
        model_status = cache.status(model_spec) if cache is not None else None
        models.append(
            {
                "id": model_spec.id,
                "title": model_spec.title,
                "cache_state": model_status.state if model_status is not None else "missing",
                "expected_bytes": model_spec.expected_bytes,
                "actual_bytes": model_status.actual_bytes if model_status is not None else None,
            }
        )

    refiners: list[JsonObject] = []
    refiner_cache = RefinerCache(refiner_cache_dir) if refiner_cache_dir is not None else None
    for refiner_spec in all_refiner_specs():
        refiner_status = refiner_cache.status(refiner_spec) if refiner_cache is not None else None
        refiners.append(
            {
                "id": refiner_spec.id,
                "title": refiner_spec.title,
                "cache_state": refiner_status.state if refiner_status is not None else "missing",
                "source_ref": refiner_spec.source_ref,
                "sam_expected_bytes": refiner_spec.sam_expected_bytes,
                "sam_actual_bytes": (
                    refiner_status.sam_actual_bytes if refiner_status is not None else None
                ),
                "mam_expected_bytes": refiner_spec.mam_expected_bytes,
                "mam_actual_bytes": (
                    refiner_status.mam_actual_bytes if refiner_status is not None else None
                ),
            }
        )

    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "torch": _torch_diagnostics(),
        "runtime": _runtime_diagnostics(runtime_cache_dir),
        "mam": {
            "source_ref": MAM_SOURCE_REF,
        },
        "models": models,
        "refiners": refiners,
    }
