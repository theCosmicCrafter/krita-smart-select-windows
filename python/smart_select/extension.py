import ctypes
import os
import platform
import shutil
import sys
from pathlib import Path

from krita import Extension


class SmartSelectExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self._library = None

    def setup(self):
        self._load_native_plugin()

    def createActions(self, window):
        pass

    def _load_native_plugin(self):
        package_dir = Path(__file__).resolve().parent
        lib_dir = package_dir / "lib"
        print(
            "[SmartSelect] event=setup.start "
            f"plugin_version=0.1.0-mvp "
            f"python_version={sys.version.split()[0]} "
            f"sys_executable={sys.executable} "
            f"platform={platform.platform()} "
            f"machine={platform.machine()} "
            f"package_dir={package_dir}",
            flush=True,
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
                python_executable = shutil.which("python3") or sys.executable
                print(
                    "[SmartSelect] event=setup.native_library.load "
                    f"library={candidate} "
                    f"worker_python={python_executable}",
                    flush=True,
                )
                self._library = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                load_with_runtime = getattr(self._library, "load_smart_select_plugin_with_runtime", None)
                if load_with_runtime:
                    load_with_runtime.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                    load_with_runtime.restype = None
                    load_with_runtime(
                        str(package_dir).encode("utf-8"),
                        str(python_executable).encode("utf-8"),
                    )
                else:
                    print("[SmartSelect] event=setup.native_abi.fallback", flush=True)
                    self._library.load_smart_select_plugin()
                print("[SmartSelect] event=setup.complete", flush=True)
                return

        names = ", ".join(str(path) for path in candidates)
        print(f"[SmartSelect] event=setup.failed reason=native_library_not_found checked={names}", flush=True)
        raise RuntimeError(f"Smart Select native library was not found. Checked: {names}")
