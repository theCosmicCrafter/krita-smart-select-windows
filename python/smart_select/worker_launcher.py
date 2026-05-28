#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

RUNTIME_WORKER_PYTHON = "3.12"


def main() -> NoReturn:
    if len(sys.argv) < 2:
        print(
            "[SmartSelectWorker] event=worker.launcher.failed reason=missing_worker_script",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    uv = shutil.which("uv")
    if uv is None:
        print(
            "[SmartSelectWorker] event=worker.launcher.failed reason=uv_not_found",
            file=sys.stderr,
            flush=True,
        )
        os.execvp("python3", ["python3", *sys.argv[1:]])

    print(
        f"[SmartSelectWorker] event=worker.launcher.exec mode=uv python={RUNTIME_WORKER_PYTHON}",
        file=sys.stderr,
        flush=True,
    )
    env = os.environ.copy()
    _configure_cuda_environment(env)
    os.execvpe(
        uv,
        [
            uv,
            "run",
            "--no-project",
            "--python",
            RUNTIME_WORKER_PYTHON,
            "python",
            *sys.argv[1:],
        ],
        env,
    )
    raise SystemExit(127)


def _configure_cuda_environment(env: dict[str, str]) -> None:
    cuda_home = _detect_cuda_home(env)
    if cuda_home is None:
        return

    env.setdefault("SMART_SELECT_CUDA_HOME", str(cuda_home))
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["PATH"] = _prepend_env_path(str(cuda_home / "bin"), env.get("PATH", ""))

    library_dirs = [
        str(candidate)
        for candidate in (cuda_home / "lib64", cuda_home / "lib")
        if candidate.exists()
    ]
    if library_dirs:
        env["LD_LIBRARY_PATH"] = _prepend_env_path(
            os.pathsep.join(library_dirs),
            env.get("LD_LIBRARY_PATH", ""),
        )


def _detect_cuda_home(env: dict[str, str]) -> Path | None:
    for key in ("SMART_SELECT_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"):
        value = env.get(key)
        if value and _is_cuda_home(Path(value)):
            return Path(value)

    for candidate in (
        Path("/usr/local/cuda-12.4"),
        Path("/usr/local/cuda"),
        Path.home() / ".local/share/krita/smart_select/cuda-12.4",
        Path.home() / "miniconda3",
    ):
        if _is_cuda_home(candidate):
            return candidate

    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc is None:
        return None
    candidate = Path(nvcc).resolve().parents[1]
    return candidate if _is_cuda_home(candidate) else None


def _is_cuda_home(path: Path) -> bool:
    return (path / "bin" / "nvcc").exists() and (
        (path / "lib64").exists() or (path / "lib").exists()
    )


def _prepend_env_path(prefix: str, current: str) -> str:
    return f"{prefix}{os.pathsep}{current}" if current else prefix


if __name__ == "__main__":
    main()
