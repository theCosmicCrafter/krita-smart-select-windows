from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobContext:
    job_id: int
    cancel_event: threading.Event = field(default_factory=threading.Event)
    temp_paths: list[Path] = field(default_factory=list)
    active_process: subprocess.Popen[bytes] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self.lock:
            process = self.active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def set_active_process(self, process: subprocess.Popen[bytes] | None) -> None:
        with self.lock:
            self.active_process = process


JobTarget = Callable[[JobContext, Mapping[str, Any]], None]
ErrorCallback = Callable[[int, str, str], None]


class JobRunner:
    def __init__(self, target: JobTarget, emit_error: ErrorCallback) -> None:
        self._target = target
        self._emit_error = emit_error
        self._lock = threading.Lock()
        self._active: JobContext | None = None

    def start(self, job_id: int, params: Mapping[str, Any]) -> bool:
        with self._lock:
            self._clear_finished_locked()
            if self._active is not None:
                self._emit_error(job_id, "WorkerBusy", "worker is busy")
                return False

            context = JobContext(job_id=job_id)
            self._active = context

        thread = threading.Thread(
            target=self._run,
            args=(context, dict(params)),
            name=f"smart-select-job-{job_id}",
            daemon=False,
        )
        thread.start()
        return True

    def cancel(self, job_id: int) -> bool:
        with self._lock:
            context = self._active
        if context is None or context.job_id != job_id:
            return False
        context.cancel()
        return True

    def _run(self, context: JobContext, params: Mapping[str, Any]) -> None:
        try:
            self._target(context, params)
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            self._emit_error(context.job_id, exc.__class__.__name__, "Smart Select worker failed")
        finally:
            with self._lock:
                if self._active is context:
                    self._active = None

    def _clear_finished_locked(self) -> None:
        return
