from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from worker.runtime_manifest import RUNTIME_PYTHON_VERSION

PYTHON_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "python" / "smart_select"
if str(PYTHON_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))

import worker_launcher  # noqa: E402

CMD_LAUNCHER = PYTHON_PACKAGE_DIR / "worker_launcher.cmd"

STUB_WORKER = """\
import json
import sys

print("stub-worker-started", sys.stdin.encoding, file=sys.stderr, flush=True)
for raw_line in sys.stdin:
    message = json.loads(raw_line)
    if message["method"] == "exit":
        reply = {"id": message["id"], "result": {"bye": True}}
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\\n")
        sys.stdout.flush()
        raise SystemExit(message["params"]["code"])
    reply = {"id": message["id"], "result": {"echo": message["params"]}}
    sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\\n")
    sys.stdout.flush()
"""


@contextmanager
def _extension_module() -> Iterator[types.ModuleType]:
    class Extension:
        def __init__(self, parent: object) -> None:
            pass

    krita = types.ModuleType("krita")
    krita.Extension = Extension  # type: ignore[attr-defined]
    with mock.patch.dict(sys.modules, {"krita": krita}):
        sys.modules.pop("extension", None)
        import extension

        yield extension
    sys.modules.pop("extension", None)


class WorkerExecutableTests(unittest.TestCase):
    def test_windows_uses_batch_entry_point(self) -> None:
        package_dir = Path("pykrita") / "smart_select"
        with _extension_module() as extension, mock.patch.object(sys, "platform", "win32"):
            with mock.patch.object(shutil, "which", return_value=None):
                without_uv = extension._worker_executable(package_dir)
            with mock.patch.object(shutil, "which", return_value="uv.exe"):
                with_uv = extension._worker_executable(package_dir)

        self.assertEqual(without_uv, str(package_dir / "worker_launcher.cmd"))
        self.assertEqual(with_uv, without_uv)

    def test_linux_behaviour_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            launcher = package_dir / "worker_launcher.py"
            launcher.write_text("", encoding="utf-8")
            with _extension_module() as extension, mock.patch.object(sys, "platform", "linux"):
                with mock.patch.object(shutil, "which", return_value="/usr/bin/uv"):
                    with_uv = extension._worker_executable(package_dir)
                with mock.patch.object(shutil, "which", return_value=None):
                    without_uv = extension._worker_executable(package_dir)

        self.assertEqual(with_uv, str(launcher))
        self.assertEqual(without_uv, sys.executable)


class PythonLauncherTests(unittest.TestCase):
    def test_windows_runs_child_and_forwards_exit_code(self) -> None:
        argv = ["uv", "run", "python", "worker.py"]
        env = {"PATH": "x"}
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(subprocess, "call", return_value=7) as call,
            mock.patch.object(os, "execvpe") as execvpe,
            self.assertRaises(SystemExit) as raised,
        ):
            worker_launcher._run(argv, env)

        self.assertEqual(raised.exception.code, 7)
        call.assert_called_once_with(argv, env=env)
        execvpe.assert_not_called()

    def test_posix_replaces_the_process(self) -> None:
        argv = ["uv", "run", "python", "worker.py"]
        env = {"PATH": "x"}
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(subprocess, "call") as call,
            mock.patch.object(os, "execvpe", side_effect=RuntimeError("exec")) as execvpe,
            self.assertRaises(RuntimeError),
        ):
            worker_launcher._run(argv, env)

        execvpe.assert_called_once_with("uv", argv, env)
        call.assert_not_called()


class BatchLauncherTests(unittest.TestCase):
    def test_batch_file_uses_crlf_line_endings(self) -> None:
        data = CMD_LAUNCHER.read_bytes()

        self.assertIn(b"\r\n", data)
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))

    def test_batch_file_matches_runtime_python_version(self) -> None:
        text = CMD_LAUNCHER.read_text(encoding="ascii")

        self.assertEqual(worker_launcher.RUNTIME_WORKER_PYTHON, RUNTIME_PYTHON_VERSION)
        self.assertRegex(
            text, rf"uv run --no-project --python {re.escape(RUNTIME_PYTHON_VERSION)} "
        )
        self.assertIn(f"runtime.python.version={RUNTIME_PYTHON_VERSION} ", text)


@unittest.skipUnless(sys.platform == "win32", "Windows entry point")
class BatchLauncherProcessTests(unittest.TestCase):
    """Start the batch file the way QProcess does: program plus one script argument."""

    def setUp(self) -> None:
        self.uv = shutil.which("uv")
        if self.uv is None:
            self.skipTest("uv is not installed")
        probe = subprocess.run(
            [self.uv, "python", "find", RUNTIME_PYTHON_VERSION], capture_output=True, check=False
        )
        if probe.returncode != 0:
            self.skipTest(f"uv has no Python {RUNTIME_PYTHON_VERSION} installed")

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        # Spaces, parentheses and non-ASCII characters, like a real Krita install path.
        package_dir = Path(temp_dir.name) / "Krita (x64) Фёдор" / "smart_select"
        (package_dir / "worker").mkdir(parents=True)
        self.launcher = package_dir / "worker_launcher.cmd"
        shutil.copyfile(CMD_LAUNCHER, self.launcher)
        self.script = package_dir / "worker" / "smart_select_worker.py"
        self.script.write_text(STUB_WORKER, encoding="utf-8")

    def start(self, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            [str(self.launcher), str(self.script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.addCleanup(process.kill)
        return process

    @staticmethod
    def request(request_id: int, method: str, params: dict[str, object]) -> bytes:
        message = {"id": request_id, "method": method, "params": params}
        return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")

    def test_ndjson_round_trip_and_exit_code(self) -> None:
        process = self.start()
        payload = self.request(1, "ping", {"path": "C:/Users/Олег/image.png"})
        payload += self.request(2, "exit", {"code": 3})

        stdout, stderr = process.communicate(payload, timeout=120)

        replies = [json.loads(line) for line in stdout.decode("utf-8").splitlines()]
        self.assertEqual(
            replies,
            [
                {"id": 1, "result": {"echo": {"path": "C:/Users/Олег/image.png"}}},
                {"id": 2, "result": {"bye": True}},
            ],
        )
        self.assertEqual(process.returncode, 3)
        log = stderr.decode("utf-8")
        self.assertIn("event.name=worker.launcher.exec", log)
        self.assertIn("stub-worker-started utf-8", log)

    def test_clean_exit_when_stdin_closes(self) -> None:
        process = self.start()

        stdout, _ = process.communicate(self.request(1, "ping", {}), timeout=120)

        self.assertEqual(json.loads(stdout), {"id": 1, "result": {"echo": {}}})
        self.assertEqual(process.returncode, 0)

    def test_reports_missing_uv(self) -> None:
        system32 = Path(os.environ["SYSTEMROOT"]) / "System32"
        process = self.start(env={"SYSTEMROOT": os.environ["SYSTEMROOT"], "PATH": str(system32)})

        stdout, stderr = process.communicate(b"", timeout=60)

        self.assertEqual(stdout, b"")
        self.assertEqual(process.returncode, 127)
        self.assertIn("error.reason=uv_not_found", stderr.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
