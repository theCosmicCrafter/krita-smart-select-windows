#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
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
        os.environ.copy(),
    )
    raise SystemExit(127)


if __name__ == "__main__":
    main()
