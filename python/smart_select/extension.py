from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from krita import Extension


def _log_event(event_name: str, **fields: object) -> None:
    parts = [
        "[SmartSelect]",
        f"event.name={_format_value(event_name)}",
        "event.domain=smart_select",
        "severity=info",
        "component=python_loader",
    ]
    parts.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


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

    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> str:
    return str(value)


class SmartSelectExtension(Extension):
    def __init__(self, parent: object) -> None:
        super().__init__(parent)
        self._library: ctypes.CDLL | None = None

    def setup(self) -> None:
        self._load_native_plugin()

    def createActions(self, window: object) -> None:
        pass

    def _load_native_plugin(self) -> None:
        package_dir = Path(__file__).resolve().parent
        lib_dir = package_dir / "lib"
        _log_event(
            "setup.start",
            **{
                "plugin.version": "0.1.0-mvp",
                "runtime.python.version": sys.version.split()[0],
                "python.executable.path": sys.executable,
                "platform.name": platform.platform(),
                "platform.machine": platform.machine(),
                "package.dir": str(package_dir),
            },
        )
        candidates = [
            lib_dir / "kritasmartselect.so",
            lib_dir / "libkritasmartselect.so",
            package_dir / "kritasmartselect.so",
            package_dir / "libkritasmartselect.so",
        ]

        if sys.platform.startswith("linux"):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)

        for candidate in candidates:
            if candidate.exists():
                python_executable = _worker_executable(package_dir)
                _log_event(
                    "setup.native_library.load",
                    **{
                        "library.path": str(candidate),
                        "python.executable.path": str(python_executable),
                    },
                )
                self._library = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                load_with_runtime = getattr(
                    self._library, "load_smart_select_plugin_with_runtime", None
                )
                if load_with_runtime:
                    load_with_runtime.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                    load_with_runtime.restype = None
                    load_with_runtime(
                        str(package_dir).encode("utf-8"),
                        str(python_executable).encode("utf-8"),
                    )
                else:
                    _log_event("setup.native_abi.fallback")
                    self._library.load_smart_select_plugin()
                _log_event("setup.complete")
                return

        names = ", ".join(str(path) for path in candidates)
        _log_event(
            "setup.failed",
            **{
                "error.reason": "native_library_not_found",
                "library.checked": names,
            },
        )
        raise RuntimeError(f"Smart Select native library was not found. Checked: {names}")


def _worker_executable(package_dir: Path) -> str:
    launcher = package_dir / "worker_launcher.py"
    if launcher.exists() and shutil.which("uv") is not None:
        return str(launcher)
    return shutil.which("python3") or sys.executable
