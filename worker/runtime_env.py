from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from job_runner import JobContext
from observability import log_event
from runtime_manifest import (
    PYTHON_PACKAGES,
    RUNTIME_PYTHON_VERSION,
    RUNTIME_VERSION,
    TORCH_INDEX_URL,
    TORCH_PACKAGES,
)

RuntimeState = Literal["missing", "ready", "invalid"]
ProgressCallback = Callable[[str, float | None, dict[str, object]], None]


class RuntimeEnvError(RuntimeError):
    pass


class RuntimeCacheDirMissing(RuntimeEnvError):
    pass


class RuntimeBootstrapError(RuntimeEnvError):
    pass


class RuntimeDependencyError(RuntimeEnvError):
    pass


class RuntimeValidationError(RuntimeEnvError):
    pass


@dataclass(frozen=True)
class RuntimeStatus:
    state: RuntimeState
    root: Path
    venv_path: Path
    python_executable: Path
    runtime_json: Path
    version: str | None
    python_version: str | None


@dataclass(frozen=True)
class RuntimeReady:
    root: Path
    venv_path: Path
    python_executable: Path
    site_packages: Path
    torch_version: str | None
    cuda_available: bool


def runtime_status(root: Path) -> RuntimeStatus:
    venv_path = root / "venv"
    python_executable = venv_path / "bin" / "python"
    runtime_json = root / "runtime.json"
    if not runtime_json.exists() or not python_executable.exists():
        return RuntimeStatus(
            "missing", root, venv_path, python_executable, runtime_json, None, None
        )

    try:
        payload = json.loads(runtime_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RuntimeStatus(
            "invalid", root, venv_path, python_executable, runtime_json, None, None
        )

    version = payload.get("runtime_version")
    python_version = payload.get("python_version")
    state: RuntimeState = (
        "ready"
        if version == RUNTIME_VERSION and _python_version_matches_runtime(python_version)
        else "invalid"
    )
    return RuntimeStatus(
        state,
        root,
        venv_path,
        python_executable,
        runtime_json,
        version if isinstance(version, str) else None,
        python_version if isinstance(python_version, str) else None,
    )


def ensure_runtime(
    root: Path,
    progress: ProgressCallback,
    context: JobContext | None = None,
) -> RuntimeReady:
    if _is_plugin_source_path(root):
        raise RuntimeCacheDirMissing("runtime cache is not writable")

    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "pip-cache").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeCacheDirMissing("runtime cache is not writable") from exc

    status = runtime_status(root)
    progress("checking_runtime", None, {"runtime_state": status.state})
    if status.state == "invalid":
        shutil.rmtree(status.venv_path, ignore_errors=True)
        try:
            status.runtime_json.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RuntimeCacheDirMissing("runtime cache is not writable") from exc
        status = runtime_status(root)

    if status.state != "ready":
        _check_cancelled(context)
        progress("creating_runtime", None, {})
        _create_venv(status.venv_path, context)
        _check_cancelled(context)
        progress("installing_runtime_dependencies", None, {})
        _install_dependencies(status.python_executable, root / "pip-cache", context)
        _check_cancelled(context)

    progress("validating_runtime", None, {})
    ready = _validate_runtime(root)
    _write_runtime_json(root, ready)
    _activate_runtime(ready)
    log_event(
        "runtime.setup.result",
        component="runtime_env",
        **{
            "runtime.cache.state": "ready",
            "runtime.version": RUNTIME_VERSION,
            "cuda.available": ready.cuda_available,
        },
    )
    return ready


def _create_venv(venv_path: Path, context: JobContext | None) -> None:
    if venv_path.exists():
        shutil.rmtree(venv_path)
    _run_process([sys.executable, "-m", "venv", str(venv_path)], {}, context)


def _install_dependencies(
    python_executable: Path, pip_cache: Path, context: JobContext | None
) -> None:
    base_env = {
        "PIP_CACHE_DIR": str(pip_cache),
        "PYTHONNOUSERSITE": "1",
    }
    _run_process(
        [str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        base_env,
        context,
    )
    _run_process(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--index-url",
            TORCH_INDEX_URL,
            *TORCH_PACKAGES,
        ],
        base_env,
        context,
    )
    _run_process(
        [str(python_executable), "-m", "pip", "install", *PYTHON_PACKAGES], base_env, context
    )


def _validate_runtime(root: Path) -> RuntimeReady:
    status = runtime_status(root)
    python_executable = status.python_executable
    site_packages = _site_packages_path(python_executable)
    if not site_packages.exists():
        raise RuntimeValidationError("alpha runtime validation failed")
    _activate_runtime_path(site_packages, python_executable)
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("PIL.Image")
        importlib.import_module("torchvision.transforms")
        transformers = importlib.import_module("transformers")
        huggingface_hub = importlib.import_module("huggingface_hub")
        if not hasattr(transformers, "AutoModelForImageSegmentation"):
            raise RuntimeDependencyError("alpha runtime dependencies are not available")
        if not hasattr(huggingface_hub, "snapshot_download"):
            raise RuntimeDependencyError("alpha runtime dependencies are not available")
        importlib.import_module("safetensors")
    except Exception as exc:
        raise RuntimeDependencyError("alpha runtime dependencies are not available") from exc
    return RuntimeReady(
        root=root,
        venv_path=status.venv_path,
        python_executable=python_executable,
        site_packages=site_packages,
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
    )


def _site_packages_path(python_executable: Path) -> Path:
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip())


def _write_runtime_json(root: Path, ready: RuntimeReady) -> None:
    payload = {
        "runtime_version": RUNTIME_VERSION,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": ready.torch_version,
        "cuda_available": ready.cuda_available,
    }
    (root / "runtime.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _activate_runtime(ready: RuntimeReady) -> None:
    _activate_runtime_path(ready.site_packages, ready.python_executable)
    os.environ["VIRTUAL_ENV"] = str(ready.venv_path)
    os.environ["PATH"] = str(ready.venv_path / "bin") + os.pathsep + os.environ.get("PATH", "")


def _activate_runtime_path(site_packages: Path, python_executable: Path) -> None:
    site_packages_text = str(site_packages)
    if site_packages_text not in sys.path:
        sys.path.insert(0, site_packages_text)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["VIRTUAL_ENV"] = str(python_executable.parent.parent)


def _run_process(
    command: Sequence[str],
    extra_env: Mapping[str, str],
    context: JobContext | None,
) -> None:
    _check_cancelled(context)
    env = os.environ.copy()
    env.update(extra_env)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if context is not None:
        context.set_active_process(process)
    output_tail: deque[str] = deque(maxlen=20)
    try:
        stdout = process.stdout
        if stdout is not None:
            try:
                for line in stdout:
                    text = line.strip()
                    if text:
                        output_tail.append(text)
            finally:
                stdout.close()
        return_code = process.wait()
    finally:
        if context is not None:
            context.set_active_process(None)
    if context is not None and context.is_cancelled():
        raise RuntimeBootstrapError("alpha runtime setup failed")
    if return_code != 0:
        log_event(
            "runtime.process.failed",
            severity="error",
            component="runtime_env",
            **{
                "process.exit_code": return_code,
                "process.argv.0": command[0] if command else "",
                "process.output.tail": " | ".join(output_tail),
                "error.type": "RuntimeBootstrapError",
            },
        )
        raise RuntimeBootstrapError("alpha runtime setup failed")


def _check_cancelled(context: JobContext | None) -> None:
    if context is not None and context.is_cancelled():
        raise RuntimeBootstrapError("runtime setup cancelled")


def _python_version_matches_runtime(value: object) -> bool:
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    return isinstance(value, str) and value == expected and value == RUNTIME_PYTHON_VERSION


def _is_plugin_source_path(root: Path) -> bool:
    try:
        worker_dir = Path(__file__).resolve().parent
        return root.resolve().is_relative_to(worker_dir.parent.resolve())
    except OSError:
        return True
