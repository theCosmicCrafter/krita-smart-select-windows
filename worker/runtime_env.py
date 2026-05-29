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
from observability import log_event
from runtime_manifest import (
    MAM_SOURCE_REF,
    MAM_SOURCE_URL,
    PYTHON_PACKAGES,
    RUNTIME_PYTHON_VERSION,
    RUNTIME_VERSION,
    SAM2_BUILD_ALLOW_ERRORS,
    SAM2_BUILD_CUDA,
    SAM2_PACKAGE,
    SAM2_REQUIRE_CUDA_EXTENSION,
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


class RefinerRuntimeValidationError(RuntimeEnvError):
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
    cuda_home: str | None
    sam2_cuda_extension: bool


@dataclass(frozen=True)
class MamSourceReady:
    source_root: Path
    source_ref: str


def _required_python_major_minor() -> tuple[int, int]:
    major, minor = RUNTIME_PYTHON_VERSION.split(".", 1)
    return int(major), int(minor)


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
    log_event(
        "runtime.setup.result",
        component="runtime_env",
        **{
            "runtime.cache.state": "ready",
            "runtime.version": RUNTIME_VERSION,
            "cuda.available": ready.cuda_available,
            "runtime.sam2.cuda_extension": ready.sam2_cuda_extension,
        },
    )
    return ready


def ensure_mam_source(
    runtime_ready: RuntimeReady,
    progress: ProgressCallback,
    context: JobContext | None = None,
) -> MamSourceReady:
    source_root = runtime_ready.root / "sources" / "matting-anything" / MAM_SOURCE_REF
    if _mam_source_ready(source_root):
        return MamSourceReady(source_root=source_root, source_ref=MAM_SOURCE_REF)

    progress("validating_runtime", None, {"runtime.source": "matting-anything"})
    _check_cancelled(context)
    source_root.parent.mkdir(parents=True, exist_ok=True)
    if source_root.exists():
        shutil.rmtree(source_root)
    git = shutil.which("git")
    if git is None:
        raise RefinerRuntimeValidationError("edge refiner runtime validation failed")
    log_event(
        "runtime.mam_source.fetch.start",
        component="runtime_env",
        **{"runtime.sam2.ref": MAM_SOURCE_REF},
    )
    _run_process([git, "clone", "--no-checkout", MAM_SOURCE_URL, str(source_root)], {}, context)
    _run_process([git, "-C", str(source_root), "checkout", MAM_SOURCE_REF], {}, context)
    if not _mam_source_ready(source_root):
        raise RefinerRuntimeValidationError("edge refiner runtime validation failed")
    log_event(
        "runtime.mam_source.fetch.result",
        component="runtime_env",
        **{"runtime.cache.state": "ready", "runtime.sam2.ref": MAM_SOURCE_REF},
    )
    return MamSourceReady(source_root=source_root, source_ref=MAM_SOURCE_REF)


def _is_plugin_source_path(root: Path) -> bool:
    try:
        worker_dir = Path(__file__).resolve().parent
        return root.resolve().is_relative_to(worker_dir.parent.resolve())
    except OSError:
        return True


def _mam_source_ready(source_root: Path) -> bool:
    if not (source_root / "networks").is_dir():
        return False
    if not (source_root / "segment-anything" / "segment_anything").is_dir():
        return False
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == MAM_SOURCE_REF


def _check_cancelled(context: JobContext | None) -> None:
    if context is not None and context.is_cancelled():
        raise RuntimeBootstrapError("runtime setup cancelled")


def _create_venv(venv_path: Path, context: JobContext | None) -> None:
    python_executable = venv_path / "bin" / "python"
    if _python_executable_matches_runtime(python_executable):
        return

    log_event("runtime.setup.start", component="runtime_env", **{"job.stage": "create_venv"})
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

    log_event(
        "runtime.setup.failed",
        severity="error",
        component="runtime_env",
        **{
            "error.reason": "compatible_python_not_found",
            "runtime.python.required_version": RUNTIME_PYTHON_VERSION,
        },
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
    log_event(
        "runtime.setup.start",
        component="runtime_env",
        **{"job.stage": "install_dependencies"},
    )
    base_env = {
        "PIP_CACHE_DIR": str(pip_cache_dir),
        "UV_CACHE_DIR": str(pip_cache_dir / "uv"),
        "PYTHONNOUSERSITE": "1",
    }
    cuda_home = _detect_cuda_home()
    if cuda_home is not None:
        base_env.update(_cuda_build_environment(cuda_home))
        log_event(
            "runtime.cuda_toolkit.found",
            component="runtime_env",
            **{"cuda.home": cuda_home},
        )
    elif SAM2_BUILD_CUDA == "1":
        log_event("runtime.cuda_toolkit.missing", severity="warning", component="runtime_env")

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
        base_env,
        context,
    )
    _run_process(
        _install_command(
            python_executable,
            (SAM2_PACKAGE,),
            force_reinstall=True,
            no_deps=True,
            no_build_isolation=True,
        ),
        {
            **base_env,
            "SAM2_BUILD_CUDA": SAM2_BUILD_CUDA,
            "SAM2_BUILD_ALLOW_ERRORS": SAM2_BUILD_ALLOW_ERRORS,
        },
        context,
    )


def _detect_cuda_home() -> str | None:
    for key in ("SMART_SELECT_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value and _is_cuda_home(Path(value)):
            return str(Path(value))

    for candidate in (
        Path("/usr/local/cuda-12.4"),
        Path("/usr/local/cuda"),
        Path.home() / ".local/share/krita/smart_select/cuda-12.4",
        Path.home() / "miniconda3",
    ):
        if _is_cuda_home(candidate):
            return str(candidate)

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        candidate = Path(nvcc).resolve().parents[1]
        if _is_cuda_home(candidate):
            return str(candidate)
    return None


def _is_cuda_home(path: Path) -> bool:
    return (path / "bin" / "nvcc").exists() and (
        (path / "lib64").exists() or (path / "lib").exists()
    )


def _cuda_build_environment(cuda_home: str) -> dict[str, str]:
    path = Path(cuda_home)
    existing_path = os.environ.get("PATH", "")
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_dirs = [
        str(candidate) for candidate in (path / "lib64", path / "lib") if candidate.exists()
    ]
    env = {
        "CUDA_HOME": cuda_home,
        "CUDA_PATH": cuda_home,
        "PATH": f"{path / 'bin'}{os.pathsep}{existing_path}"
        if existing_path
        else str(path / "bin"),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [*library_dirs, existing_library_path] if existing_library_path else library_dirs
        ),
    }
    conda_cc = path / "bin" / "x86_64-conda-linux-gnu-gcc"
    conda_cxx = path / "bin" / "x86_64-conda-linux-gnu-g++"
    if conda_cc.exists() and conda_cxx.exists():
        env["CC"] = str(conda_cc)
        env["CXX"] = str(conda_cxx)
        env["CUDAHOSTCXX"] = str(conda_cxx)
    return env


def _install_command(
    python_executable: Path,
    packages: Sequence[str],
    *,
    index_url: str | None = None,
    upgrade: bool = False,
    force_reinstall: bool = False,
    no_deps: bool = False,
    no_build_isolation: bool = False,
) -> list[str]:
    if _python_has_pip(python_executable):
        command = [str(python_executable), "-m", "pip", "install"]
        if upgrade:
            command.append("--upgrade")
        if force_reinstall:
            command.append("--force-reinstall")
        if no_deps:
            command.append("--no-deps")
        if no_build_isolation:
            command.append("--no-build-isolation")
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
    if force_reinstall:
        command.append("--force-reinstall")
    if no_deps:
        command.append("--no-deps")
    if no_build_isolation:
        command.append("--no-build-isolation")
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
                log_event(
                    "runtime.setup.progress",
                    component="runtime_env",
                    **{"process.output.line": line},
                )
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
        sam2_cuda_extension = _sam2_cuda_extension_available()
    except Exception as exc:
        log_event(
            "runtime.validation.failed",
            severity="error",
            component="runtime_env",
            **{"error.type": type(exc).__name__, "error.reason": str(exc)},
        )
        raise RuntimeDependencyError("SAM2 dependencies are not available") from exc
    if SAM2_REQUIRE_CUDA_EXTENSION and not sam2_cuda_extension:
        log_event(
            "runtime.validation.failed",
            severity="error",
            component="runtime_env",
            **{
                "error.type": "Sam2CudaExtensionMissing",
                "error.reason": "sam2._C import failed",
            },
        )
        raise RuntimeDependencyError("SAM2 dependencies are not available") from None

    version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    cuda_home = _detect_cuda_home()
    return RuntimeReady(
        root=root,
        venv_path=status.venv_path,
        python_executable=status.python_executable,
        site_packages=site_packages,
        torch_version=str(version) if version is not None else None,
        cuda_available=cuda_available,
        cuda_home=cuda_home,
        sam2_cuda_extension=sam2_cuda_extension,
    )


def _sam2_cuda_extension_available() -> bool:
    try:
        importlib.import_module("sam2._C")
    except Exception as exc:
        log_event(
            "runtime.validation.sam2_cuda_extension_failed",
            severity="warning",
            component="runtime_env",
            **{"error.type": type(exc).__name__, "error.reason": str(exc)},
        )
        return False
    return True


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
        "cuda_home": ready.cuda_home,
        "sam2_cuda_extension": ready.sam2_cuda_extension,
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
