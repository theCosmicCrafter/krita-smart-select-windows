from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from job_runner import JobContext
from runtime_manifest import (
    PYTHON_PACKAGES,
    RUNTIME_PYTHON_VERSION,
    RUNTIME_VERSION,
    SAM2_BUILD_CUDA,
    SAM2_SOURCE_REF,
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
    sam2_source_ref: str | None
    python_version: str | None


@dataclass(frozen=True)
class RuntimeReady:
    root: Path
    venv_path: Path
    python_executable: Path
    site_packages: Path
    torch_version: str | None
    cuda_available: bool


def _required_python_major_minor() -> tuple[int, int]:
    major, minor = RUNTIME_PYTHON_VERSION.split(".", 1)
    return int(major), int(minor)


def _log_worker_event(event: str, **fields: object) -> None:
    parts = [f"[SmartSelectWorker] event={event}"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), file=sys.stderr, flush=True)


def runtime_status(root: Path) -> RuntimeStatus:
    venv_path = root / "venv"
    python_executable = venv_path / "bin" / "python"
    runtime_json = root / "runtime.json"
    if not runtime_json.exists() or not python_executable.exists():
        return RuntimeStatus(
            state="missing",
            root=root,
            venv_path=venv_path,
            python_executable=python_executable,
            runtime_json=runtime_json,
            version=None,
            sam2_source_ref=None,
            python_version=None,
        )

    try:
        payload = json.loads(runtime_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RuntimeStatus(
            state="invalid",
            root=root,
            venv_path=venv_path,
            python_executable=python_executable,
            runtime_json=runtime_json,
            version=None,
            sam2_source_ref=None,
            python_version=None,
        )

    version = payload.get("runtime_version")
    source_ref = payload.get("sam2_source_ref")
    python_version = payload.get("python_version")
    state: RuntimeState = (
        "ready"
        if (
            version == RUNTIME_VERSION
            and source_ref == SAM2_SOURCE_REF
            and _python_version_matches_runtime(python_version)
        )
        else "invalid"
    )
    return RuntimeStatus(
        state=state,
        root=root,
        venv_path=venv_path,
        python_executable=python_executable,
        runtime_json=runtime_json,
        version=version if isinstance(version, str) else None,
        sam2_source_ref=source_ref if isinstance(source_ref, str) else None,
        python_version=python_version if isinstance(python_version, str) else None,
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

    if not os.access(root, os.W_OK):
        raise RuntimeCacheDirMissing("runtime cache is not writable")

    status = runtime_status(root)
    progress("checking_runtime", None, {"runtime_state": status.state})
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
    _log_worker_event(
        "runtime.setup.result",
        cache_state="ready",
        runtime_version=RUNTIME_VERSION,
        cuda_available=ready.cuda_available,
    )
    return ready


def _is_plugin_source_path(root: Path) -> bool:
    try:
        worker_dir = Path(__file__).resolve().parent
        return root.resolve().is_relative_to(worker_dir.parent.resolve())
    except OSError:
        return True


def _check_cancelled(context: JobContext | None) -> None:
    if context is not None and context.is_cancelled():
        raise RuntimeBootstrapError("runtime setup cancelled")


def _create_venv(venv_path: Path, context: JobContext | None) -> None:
    python_executable = venv_path / "bin" / "python"
    if _python_executable_matches_runtime(python_executable):
        return

    _log_worker_event("runtime.setup.start", stage="create_venv")
    uv = shutil.which("uv")
    if uv is not None:
        _run_process(
            [uv, "venv", "--python", RUNTIME_PYTHON_VERSION, "--clear", str(venv_path)],
            {},
            context,
        )
        return

    if _python_executable_matches_runtime(Path(sys.executable)):
        _run_process([sys.executable, "-m", "venv", "--clear", str(venv_path)], {}, context)
        return

    _log_worker_event(
        "runtime.setup.failed",
        reason="compatible_python_not_found",
        required_python=RUNTIME_PYTHON_VERSION,
    )
    raise RuntimeBootstrapError("SAM2 runtime setup failed")


def _python_version_matches_runtime(version: object) -> bool:
    if not isinstance(version, str):
        return False
    parts = version.split(".")
    if len(parts) < 2:
        return False
    try:
        return (int(parts[0]), int(parts[1])) == _required_python_major_minor()
    except ValueError:
        return False


def _python_executable_matches_runtime(python_executable: Path) -> bool:
    return _python_major_minor(python_executable) == _required_python_major_minor()


def _python_major_minor(python_executable: Path) -> tuple[int, int] | None:
    if not python_executable.exists():
        return None
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    parts = completed.stdout.strip().split(".")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _install_dependencies(
    python_executable: Path,
    pip_cache_dir: Path,
    context: JobContext | None,
) -> None:
    _log_worker_event("runtime.setup.start", stage="install_dependencies")
    base_env = {
        "PIP_CACHE_DIR": str(pip_cache_dir),
        "UV_CACHE_DIR": str(pip_cache_dir / "uv"),
        "PYTHONNOUSERSITE": "1",
    }
    _run_process(
        _install_command(
            python_executable,
            ("pip", "setuptools", "wheel"),
            upgrade=True,
        ),
        base_env,
        context,
    )
    _run_process(
        _install_command(python_executable, TORCH_PACKAGES, index_url=TORCH_INDEX_URL),
        base_env,
        context,
    )
    _run_process(
        _install_command(python_executable, PYTHON_PACKAGES),
        {**base_env, "SAM2_BUILD_CUDA": SAM2_BUILD_CUDA},
        context,
    )


def _install_command(
    python_executable: Path,
    packages: Sequence[str],
    *,
    index_url: str | None = None,
    upgrade: bool = False,
) -> list[str]:
    if _python_has_pip(python_executable):
        command = [str(python_executable), "-m", "pip", "install"]
        if upgrade:
            command.append("--upgrade")
        if index_url is not None:
            command.extend(["--index-url", index_url])
        command.extend(packages)
        return command

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeBootstrapError("SAM2 runtime setup failed")

    command = [uv, "pip", "install", "--python", str(python_executable)]
    if upgrade:
        command.append("--upgrade")
    if index_url is not None:
        command.extend(["--index-url", index_url])
    command.extend(packages)
    return command


def _python_has_pip(python_executable: Path) -> bool:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _run_process(
    args: Sequence[str],
    extra_env: Mapping[str, str],
    context: JobContext | None,
) -> None:
    env = os.environ.copy()
    env.update(extra_env)
    process = subprocess.Popen(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if context is not None:
        context.set_active_process(process)
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                _log_worker_event("runtime.setup.progress", line=line)
            if context is not None and context.is_cancelled():
                process.terminate()
                raise RuntimeBootstrapError("runtime setup cancelled")
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeBootstrapError("SAM2 runtime setup failed")
    finally:
        if context is not None:
            context.set_active_process(None)


def _validate_runtime(root: Path) -> RuntimeReady:
    status = runtime_status(root)
    if not _python_executable_matches_runtime(Path(sys.executable)):
        raise RuntimeValidationError("SAM2 runtime validation failed")

    site_packages = _site_packages_path(status.python_executable)
    if not site_packages.exists():
        raise RuntimeValidationError("SAM2 runtime validation failed")
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))

    try:
        torch = importlib.import_module("torch")
        importlib.import_module("PIL.Image")
        importlib.import_module("sam2.build_sam")
        importlib.import_module("sam2.automatic_mask_generator")
    except Exception as exc:
        raise RuntimeDependencyError("SAM2 dependencies are not available") from exc

    version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    return RuntimeReady(
        root=root,
        venv_path=status.venv_path,
        python_executable=status.python_executable,
        site_packages=site_packages,
        torch_version=str(version) if version is not None else None,
        cuda_available=cuda_available,
    )


def _site_packages_path(python_executable: Path) -> Path:
    if python_executable.exists():
        completed = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            path = completed.stdout.strip()
            if path:
                return Path(path)

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return python_executable.parent.parent / "lib" / version / "site-packages"


def _write_runtime_json(root: Path, ready: RuntimeReady) -> None:
    payload = {
        "runtime_version": RUNTIME_VERSION,
        "python_version": _python_version_text(ready.python_executable),
        "sam2_source_ref": SAM2_SOURCE_REF,
        "torch_version": ready.torch_version,
        "cuda_available": ready.cuda_available,
    }
    (root / "runtime.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _python_version_text(python_executable: Path) -> str:
    completed = subprocess.run(
        [str(python_executable), "-c", "import platform; print(platform.python_version())"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return sys.version.split()[0]


def _activate_runtime(ready: RuntimeReady) -> None:
    if str(ready.site_packages) not in sys.path:
        sys.path.insert(0, str(ready.site_packages))
    os.environ["VIRTUAL_ENV"] = str(ready.venv_path)
    os.environ["PATH"] = str(ready.venv_path / "bin") + os.pathsep + os.environ.get("PATH", "")
