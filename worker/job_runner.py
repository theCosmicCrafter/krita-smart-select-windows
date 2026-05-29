from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability import log_event


@dataclass
class JobContext:
    job_id: int
    job_type: str = "unknown"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    temp_paths: list[Path] = field(default_factory=list)
    active_process: subprocess.Popen[Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self.lock:
            process = self.active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def set_active_process(self, process: subprocess.Popen[Any] | None) -> None:
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
        self._active_thread: threading.Thread | None = None

    def start(self, job_id: int, params: Mapping[str, Any]) -> bool:
        with self._lock:
            self._clear_finished_locked()
            if self._active is not None:
                self._emit_error(job_id, "WorkerBusy", "worker is busy")
                return False

            context = JobContext(job_id=job_id, job_type=_job_type_for_params(params))
            self._active = context

        thread = threading.Thread(
            target=self._run,
            args=(context, dict(params)),
            name=f"smart-select-job-{job_id}",
            daemon=False,
        )
        with self._lock:
            self._active_thread = thread
        thread.start()
        return True

    def cancel(self, job_id: int) -> bool:
        with self._lock:
            context = self._active
        if context is None or context.job_id != job_id:
            return False
        context.cancel()
        return True

    def wait_for_idle(self) -> None:
        while True:
            with self._lock:
                thread = self._active_thread
            if thread is None:
                return
            thread.join()

    def _run(self, context: JobContext, params: Mapping[str, Any]) -> None:
        try:
            self._target(context, params)
        except Exception as exc:
            log_event(
                "worker.job.failed",
                severity="error",
                component="worker",
                **{
                    "job.id": context.job_id,
                    "job.type": context.job_type,
                    "error.type": exc.__class__.__name__,
                    "exception.type": exc.__class__.__name__,
                    "exception.message": str(exc),
                },
            )
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            self._emit_error(context.job_id, exc.__class__.__name__, "Smart Select worker failed")
        finally:
            with self._lock:
                if self._active is context:
                    self._active = None
                    self._active_thread = None

    def _clear_finished_locked(self) -> None:
        return


def _job_type_for_params(params: Mapping[str, Any]) -> str:
    if params.get("_method") == "prepare_selection":
        return "lasso_selection"
    if params.get("_method") == "diagnostics":
        return "diagnostics"
    return "unknown"
