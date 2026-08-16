#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

RUNTIME_WORKER_PYTHON = "3.12"


def format_log_event(event_name: str, **fields: object) -> str:
    parts = [
        "[SmartSelectWorker]",
        f"event.name={_format_value(event_name)}",
        "event.domain=smart_select",
        "severity=info",
        "component=worker_launcher",
    ]
    parts.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
    return " ".join(parts)


def _format_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if text and all(ch.isalnum() or ch in "_./:@+-" for ch in text):
        return text
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def main() -> NoReturn:
    if len(sys.argv) < 2:
        print(
            format_log_event("worker.launcher.failed", **{"error.reason": "missing_worker_script"}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    uv = shutil.which("uv")
    if uv is None:
        print(
            format_log_event("worker.launcher.failed", **{"error.reason": "uv_not_found"}),
            file=sys.stderr,
            flush=True,
        )
        # On Windows there is no python3, use python or sys.executable
        python_bin = shutil.which("python3") or shutil.which("python") or sys.executable
        os.execvp(python_bin, [python_bin, *sys.argv[1:]])

    print(
        format_log_event(
            "worker.launcher.exec",
            **{
                "launcher.mode": "uv",
                "runtime.python.version": RUNTIME_WORKER_PYTHON,
            },
        ),
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

    if sys.platform.startswith("win"):
        # Windows CUDA is typically installed under Program Files
        program_files = env.get("ProgramFiles", r"C:\Program Files")
        for candidate in (
            Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.4",
            Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.6",
            Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v13.0",
            Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v13.2",
            Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v13.3",
        ):
            if _is_cuda_home(candidate):
                return candidate
    else:
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
    if sys.platform.startswith("win"):
        # On Windows, nvcc is nvcc.exe and libs are under lib/x64
        return (path / "bin" / "nvcc.exe").exists() and (
            (path / "lib" / "x64").exists() or (path / "lib").exists()
        )
    return (path / "bin" / "nvcc").exists() and (
        (path / "lib64").exists() or (path / "lib").exists()
    )


def _prepend_env_path(prefix: str, current: str) -> str:
    return f"{prefix}{os.pathsep}{current}" if current else prefix


if __name__ == "__main__":
    main()
