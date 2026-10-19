from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import job_runner  # noqa: E402
from worker import runtime_env  # noqa: E402


class _FakeProcess:
    def __init__(self, pid: int = 4242, alive: bool = True) -> None:
        self.pid = pid
        self.alive = alive
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return None if self.alive else 0

    def terminate(self) -> None:
        self.calls.append("terminate")
        self.alive = False

    def kill(self) -> None:
        self.calls.append("kill")
        self.alive = False


@contextlib.contextmanager
def _captured_stderr() -> Iterator[io.StringIO]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        yield stderr


class CancelProcessTests(unittest.TestCase):
    def test_windows_terminates_the_whole_process_tree(self) -> None:
        process = _FakeProcess()
        commands: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            commands.append(command)
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            process.alive = False
            return subprocess.CompletedProcess(command, 0, b"", b"")

        context = job_runner.JobContext(job_id=7, job_type="lasso_selection")
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(job_runner.subprocess, "run", fake_run),
            mock.patch.dict(os.environ, {"SystemRoot": r"C:\Windows"}),
            _captured_stderr() as stderr,
        ):
            context.set_active_process(process)  # type: ignore[arg-type]
            context.cancel()

        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0][0].lower().endswith("taskkill.exe"))
        self.assertEqual(commands[0][1:], ["/PID", "4242", "/T", "/F"])
        self.assertEqual(process.calls, [])
        log = stderr.getvalue()
        self.assertIn("event.name=worker.job.process.terminated", log)
        self.assertIn("job.id=7", log)
        self.assertIn("process.pid=4242", log)
        self.assertIn("process.terminate.method=taskkill_tree", log)
        self.assertIn("process.terminate.result=0", log)

    def test_windows_kills_the_process_when_taskkill_is_unavailable(self) -> None:
        process = _FakeProcess()
        context = job_runner.JobContext(job_id=1)
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(job_runner.subprocess, "run", side_effect=OSError("missing")),
            _captured_stderr() as stderr,
        ):
            context.set_active_process(process)  # type: ignore[arg-type]
            context.cancel()

        self.assertEqual(process.calls, ["kill"])
        self.assertIn("process.terminate.result=null", stderr.getvalue())

    def test_posix_terminates_only_the_process(self) -> None:
        process = _FakeProcess()
        context = job_runner.JobContext(job_id=1)
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(job_runner.subprocess, "run") as run,
            _captured_stderr() as stderr,
        ):
            context.set_active_process(process)  # type: ignore[arg-type]
            context.cancel()

        run.assert_not_called()
        self.assertEqual(process.calls, ["terminate"])
        self.assertIn("process.terminate.method=terminate", stderr.getvalue())

    def test_exited_process_is_left_alone(self) -> None:
        process = _FakeProcess(alive=False)
        context = job_runner.JobContext(job_id=1)
        with mock.patch.object(job_runner.subprocess, "run") as run, _captured_stderr() as stderr:
            context.set_active_process(process)  # type: ignore[arg-type]
            context.cancel()

        run.assert_not_called()
        self.assertEqual(process.calls, [])
        self.assertEqual(stderr.getvalue(), "")

    def test_process_registered_after_cancel_is_stopped(self) -> None:
        process = _FakeProcess()
        context = job_runner.JobContext(job_id=1)
        with mock.patch.object(sys, "platform", "linux"), _captured_stderr():
            context.cancel()
            context.set_active_process(process)  # type: ignore[arg-type]

        self.assertTrue(context.is_cancelled())
        self.assertEqual(process.calls, ["terminate"])

    def test_clearing_the_active_process_after_cancel_is_harmless(self) -> None:
        context = job_runner.JobContext(job_id=1)
        context.cancel()
        context.set_active_process(None)

        self.assertIsNone(context.active_process)


PARENT_SCRIPT = textwrap.dedent(
    """
    import json, os, subprocess, sys

    # Like the venv python.exe stub: starts the real interpreter as a child that
    # inherits the output pipe, then waits for it.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump({"parent": os.getpid(), "child": child.pid}, handle)
    child.wait()
    """
)


def _pid_alive(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        return bool(exit_code.value == still_active)
    finally:
        kernel32.CloseHandle(handle)


def _wait_until(condition: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return bool(condition())


HOLD_STDIN_SCRIPT = textwrap.dedent(
    """
    import sys, threading
    sys.path.insert(0, sys.argv[1])
    import runtime_env

    def job():
        runtime_env._run_process([sys.executable, "-c", "print('child ok')"], {}, None)
        print("job done", flush=True)

    threading.Thread(target=job).start()
    sys.stdin.readline()   # like the worker main loop, blocked on the protocol stdin
    """
)


@unittest.skipUnless(sys.platform == "win32", "stdin pipe sharing is a Windows problem")
class HelperProcessStdinTests(unittest.TestCase):
    def test_helper_process_starts_while_main_thread_waits_on_stdin(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", HOLD_STDIN_SCRIPT, str(WORKER_DIR)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(process.kill)
        lines: list[str] = []
        reader = threading.Thread(target=lambda: lines.extend(process.stdout or []))
        reader.start()

        self.assertTrue(
            _wait_until(lambda: "job done\n" in lines, 30),
            "a child of _run_process hung while the parent waited on stdin",
        )
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=10)
        reader.join(timeout=10)


@unittest.skipUnless(sys.platform == "win32", "process trees need taskkill on Windows")
class CancelProcessTreeTests(unittest.TestCase):
    """A real parent -> child chain like `venv\\Scripts\\python.exe -m pip` on Windows."""

    def test_cancel_stops_run_process_and_all_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "pids.json"
            context = job_runner.JobContext(job_id=1)
            outcome: list[BaseException | None] = []

            def run() -> None:
                try:
                    runtime_env._run_process(
                        [sys.executable, "-c", PARENT_SCRIPT, str(pid_file)], {}, context
                    )
                except BaseException as exc:
                    outcome.append(exc)
                else:
                    outcome.append(None)

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(_wait_until(pid_file.exists, 30), "child process was not started")
            self.assertTrue(
                _wait_until(lambda: pid_file.read_text(encoding="utf-8") != "", 10),
                "pid file stayed empty",
            )
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
            self.addCleanup(self._kill_leftovers, pids.values())
            self.assertTrue(_pid_alive(pids["parent"]))
            self.assertTrue(_pid_alive(pids["child"]))

            with _captured_stderr() as stderr:
                context.cancel()
                thread.join(timeout=20)

            self.assertFalse(thread.is_alive(), "_run_process did not return after cancel")
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], runtime_env.RuntimeBootstrapError)
            self.assertEqual(str(outcome[0]), "runtime setup cancelled")
            self.assertTrue(_wait_until(lambda: not _pid_alive(pids["parent"]), 5))
            self.assertTrue(_wait_until(lambda: not _pid_alive(pids["child"]), 5))
            self.assertIn("process.terminate.method=taskkill_tree", stderr.getvalue())

    @staticmethod
    def _kill_leftovers(pids: Any) -> None:
        for pid in pids:
            if _pid_alive(pid):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False
                )


if __name__ == "__main__":
    unittest.main()
