from __future__ import annotations

import importlib.util
import json
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from alpha_model_cache import ModelCache
from alpha_model_registry import all_model_specs
from runtime_env import runtime_status
from runtime_manifest import RUNTIME_VERSION

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
            "torch_importable": False,
            "birefnet_importable": False,
        }

    status = runtime_status(runtime_cache_dir)
    marker = _runtime_marker(status.runtime_json)
    return {
        "cache_state": status.state,
        "runtime_version": status.version or RUNTIME_VERSION,
        "venv_exists": status.venv_path.exists(),
        "python_executable": str(status.python_executable),
        "torch_importable": False,
        "birefnet_importable": False,
        "cuda_available": marker.get("cuda_available", False),
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
) -> JsonObject:
    models: list[JsonObject] = []
    cache = ModelCache(model_cache_dir) if model_cache_dir is not None else None
    for model_spec in all_model_specs():
        model_status = cache.status(model_spec) if cache is not None else None
        models.append(
            {
                "id": model_spec.id,
                "title": model_spec.title,
                "cache_state": model_status.state if model_status is not None else "missing",
                "checkpoint_expected_bytes": model_spec.checkpoint_expected_bytes,
                "checkpoint_actual_bytes": (
                    model_status.checkpoint_bytes if model_status is not None else None
                ),
                "repo_id": model_spec.repo_id,
                "revision": model_spec.revision,
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
