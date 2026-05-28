from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from model_cache import ModelCache
from model_registry import all_model_specs
from runtime_env import runtime_status
from runtime_manifest import RUNTIME_VERSION, SAM2_SOURCE_REF

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
    return {
        "cache_state": status.state,
        "runtime_version": status.version or RUNTIME_VERSION,
        "venv_exists": status.venv_path.exists(),
        "python_executable": str(status.python_executable),
        "sam2_source_ref": status.sam2_source_ref or SAM2_SOURCE_REF,
        "sam2_importable": False,
        "torch_importable": False,
    }


def collect_runtime_diagnostics(
    model_cache_dir: Path | None, runtime_cache_dir: Path | None = None
) -> JsonObject:
    models: list[JsonObject] = []
    cache = ModelCache(model_cache_dir) if model_cache_dir is not None else None
    for spec in all_model_specs():
        if spec.id == "mock":
            continue
        status = cache.status(spec) if cache is not None else None
        models.append(
            {
                "id": spec.id,
                "title": spec.title,
                "cache_state": status.state if status is not None else "missing",
                "expected_bytes": spec.expected_bytes,
                "actual_bytes": status.actual_bytes if status is not None else None,
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
        "models": models,
    }
