from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.runtime_manifest import (  # noqa: E402
    RUNTIME_PYTHON_VERSION,
    RUNTIME_VERSION,
    TORCH_INDEX_URL,
    TORCH_INDEX_URL_BLACKWELL,
)

from worker import runtime_env  # noqa: E402


def _no_gpu_probe() -> Any:
    """Tests must not depend on the GPU of the machine they run on."""
    return mock.patch.object(runtime_env, "detect_max_compute_capability", return_value=None)


class RuntimeEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = _no_gpu_probe()
        probe.start()
        self.addCleanup(probe.stop)

    def test_runtime_status_reports_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = runtime_env.venv_python(root / "venv")
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": RUNTIME_VERSION,
                        "python_version": RUNTIME_PYTHON_VERSION,
                    }
                ),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "ready")
        self.assertEqual(status.version, RUNTIME_VERSION)

    def test_runtime_status_reports_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = runtime_env.venv_python(root / "venv")
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": "old-runtime",
                        "python_version": RUNTIME_PYTHON_VERSION,
                    }
                ),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "invalid")

    def test_ensure_runtime_reuses_ready_runtime(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = runtime_env.venv_python(root / "venv")
            site_packages = root / "venv" / "lib" / "python3.12" / "site-packages"
            python_path.parent.mkdir(parents=True)
            site_packages.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": RUNTIME_VERSION,
                        "python_version": RUNTIME_PYTHON_VERSION,
                    }
                ),
                encoding="utf-8",
            )

            original_create = runtime_env._create_venv
            original_install = runtime_env._install_dependencies
            original_validate = runtime_env._validate_runtime
            try:
                runtime_env._create_venv = lambda *_args, **_kwargs: calls.append("create")
                runtime_env._install_dependencies = lambda *_args, **_kwargs: calls.append(
                    "install"
                )
                runtime_env._validate_runtime = lambda _root: runtime_env.RuntimeReady(
                    root=root,
                    venv_path=root / "venv",
                    python_executable=python_path,
                    site_packages=site_packages,
                    torch_version="2.6.0",
                    cuda_available=True,
                )

                ready = runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)
            finally:
                runtime_env._create_venv = original_create
                runtime_env._install_dependencies = original_install
                runtime_env._validate_runtime = original_validate

        self.assertEqual(calls, [])
        self.assertTrue(ready.cuda_available)

    def test_runtime_version_mismatch_recreates_runtime(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_venv = root / "venv"
            old_venv.mkdir(parents=True)
            (old_venv / "stale.txt").write_text("old", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps({"runtime_version": "sam2-runtime-v3", "python_version": "3.12"}),
                encoding="utf-8",
            )

            original_create = runtime_env._create_venv
            original_install = runtime_env._install_dependencies
            original_validate = runtime_env._validate_runtime
            try:
                runtime_env._create_venv = lambda *_args, **_kwargs: calls.append("create")
                runtime_env._install_dependencies = lambda *_args, **_kwargs: calls.append(
                    "install"
                )
                runtime_env._validate_runtime = lambda _root: runtime_env.RuntimeReady(
                    root=root,
                    venv_path=root / "venv",
                    python_executable=runtime_env.venv_python(root / "venv"),
                    site_packages=root / "venv" / "site-packages",
                    torch_version="2.6.0",
                    cuda_available=False,
                )

                runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)
            finally:
                runtime_env._create_venv = original_create
                runtime_env._install_dependencies = original_install
                runtime_env._validate_runtime = original_validate

        self.assertEqual(calls, ["create", "install"])
        self.assertFalse((old_venv / "stale.txt").exists())

    def test_ensure_runtime_rejects_plugin_source_path(self) -> None:
        root = Path(__file__).resolve().parents[2] / "worker" / ".runtime-test"

        with self.assertRaises(runtime_env.RuntimeCacheDirMissing):
            runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)

    def test_runtime_bootstrap_failure_maps_to_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_create = runtime_env._create_venv
            try:
                runtime_env._create_venv = self._raise_bootstrap

                with self.assertRaises(runtime_env.RuntimeBootstrapError):
                    runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)
            finally:
                runtime_env._create_venv = original_create

    def test_runtime_subprocess_output_does_not_leak_to_protocol_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runtime_env._run_process(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "print('this would corrupt ndjson'); "
                        "print('stderr is captured too', file=sys.stderr)"
                    ),
                ],
                {},
                None,
            )

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    @staticmethod
    def _raise_bootstrap(*_args: Any, **_kwargs: Any) -> None:
        raise runtime_env.RuntimeBootstrapError("alpha runtime setup failed")


class VenvLayoutTests(unittest.TestCase):
    """The managed venv is created by `python -m venv`, whose layout depends on the OS."""

    def setUp(self) -> None:
        probe = _no_gpu_probe()
        probe.start()
        self.addCleanup(probe.stop)

    def test_posix_layout(self) -> None:
        venv = Path("runtime") / "venv"

        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(runtime_env.venv_scripts_dir(venv), venv / "bin")
            self.assertEqual(runtime_env.venv_python(venv), venv / "bin" / "python")

    def test_windows_layout(self) -> None:
        venv = Path("runtime") / "venv"

        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(runtime_env.venv_scripts_dir(venv), venv / "Scripts")
            self.assertEqual(runtime_env.venv_python(venv), venv / "Scripts" / "python.exe")

    def test_runtime_status_finds_windows_venv_python(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(runtime_env, "_python_version_matches_runtime", return_value=True),
        ):
            root = Path(temp_dir)
            python_path = root / "venv" / "Scripts" / "python.exe"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps({"runtime_version": RUNTIME_VERSION, "python_version": "3.12"}),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "ready")
        self.assertEqual(status.python_executable, python_path)

    def test_runtime_status_ignores_posix_layout_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(sys, "platform", "win32"):
            root = Path(temp_dir)
            posix_python = root / "venv" / "bin" / "python"
            posix_python.parent.mkdir(parents=True)
            posix_python.write_text("", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps({"runtime_version": RUNTIME_VERSION, "python_version": "3.12"}),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "missing")

    def test_bootstrap_on_windows_uses_scripts_python_and_path(self) -> None:
        commands: list[list[str]] = []

        def fake_run_process(
            command: Sequence[str], extra_env: Mapping[str, str], context: object
        ) -> None:
            commands.append(list(command))
            if command[1:3] == ["-m", "venv"]:
                scripts = Path(command[3]) / "Scripts"
                scripts.mkdir(parents=True)
                (scripts / "python.exe").write_text("", encoding="utf-8")

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.dict(os.environ),
            mock.patch.object(runtime_env, "_run_process", fake_run_process),
        ):
            root = Path(temp_dir) / "runtime"
            venv = root / "venv"
            scripts = venv / "Scripts"
            ready = runtime_env.RuntimeReady(
                root=root,
                venv_path=venv,
                python_executable=scripts / "python.exe",
                site_packages=venv / "Lib" / "site-packages",
                torch_version="2.6.0",
                cuda_available=True,
            )
            with mock.patch.object(runtime_env, "_validate_runtime", return_value=ready):
                runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)

            self.assertEqual(commands[0], [sys.executable, "-m", "venv", str(venv)])
            pip_commands = commands[1:]
            self.assertEqual(len(pip_commands), 3)
            for command in pip_commands:
                self.assertEqual(command[0], str(scripts / "python.exe"))
                self.assertEqual(command[1:3], ["-m", "pip"])
            self.assertTrue(os.environ["PATH"].startswith(str(scripts) + os.pathsep))
            self.assertEqual(os.environ["VIRTUAL_ENV"], str(venv))
            self.assertEqual(sys.path[0], str(venv / "Lib" / "site-packages"))


class ComputeCapabilityTests(unittest.TestCase):
    def test_parses_highest_capability(self) -> None:
        parse = runtime_env._parse_max_compute_capability

        self.assertEqual(parse("12.0\n"), (12, 0))
        self.assertEqual(parse("8.6\r\n8.9\r\n"), (8, 9))
        self.assertEqual(parse("8.9\n12.0\n"), (12, 0))
        self.assertEqual(parse("9.0\n10.0\n"), (10, 0))

    def test_unparsable_output_means_unknown(self) -> None:
        parse = runtime_env._parse_max_compute_capability

        self.assertIsNone(parse(""))
        self.assertIsNone(parse("[N/A]\n"))
        self.assertIsNone(parse('Field "compute_cap" is not a valid field.\n'))

    def test_detect_runs_nvidia_smi_and_parses_output(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="12.0\n", stderr="")
        with (
            mock.patch.object(runtime_env.shutil, "which", return_value="nvidia-smi"),
            mock.patch.object(runtime_env.subprocess, "run", return_value=completed) as run,
        ):
            capability = runtime_env.detect_max_compute_capability()

        self.assertEqual(capability, (12, 0))
        self.assertEqual(
            run.call_args.args[0],
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        )

    def test_helper_processes_do_not_inherit_the_protocol_stdin(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="12.0\n", stderr="")
        with (
            mock.patch.object(runtime_env.shutil, "which", return_value="nvidia-smi"),
            mock.patch.object(runtime_env.subprocess, "run", return_value=completed) as run,
        ):
            runtime_env.detect_max_compute_capability()
            runtime_env._site_packages_path(Path("python"))
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIs(call.kwargs["stdin"], subprocess.DEVNULL)

        process = mock.MagicMock()
        process.stdout = mock.MagicMock()
        process.stdout.__iter__.return_value = iter(())
        process.wait.return_value = 0
        with mock.patch.object(runtime_env.subprocess, "Popen", return_value=process) as popen:
            runtime_env._run_process(["python", "-c", "pass"], {}, None)
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_detect_tolerates_missing_or_failing_nvidia_smi(self) -> None:
        failed = subprocess.CompletedProcess([], 2, stdout="", stderr="invalid field")
        with mock.patch.object(runtime_env.shutil, "which", return_value=None):
            self.assertIsNone(runtime_env.detect_max_compute_capability())
        for outcome in (
            {"return_value": failed},
            {"side_effect": OSError("denied")},
            {"side_effect": subprocess.TimeoutExpired("nvidia-smi", 15)},
        ):
            with (
                mock.patch.object(runtime_env.shutil, "which", return_value="nvidia-smi"),
                mock.patch.object(runtime_env.subprocess, "run", **outcome),
            ):
                self.assertIsNone(runtime_env.detect_max_compute_capability())


class TorchIndexTests(unittest.TestCase):
    default_index = TORCH_INDEX_URL
    blackwell_index = TORCH_INDEX_URL_BLACKWELL

    def select(self, capability: tuple[int, int] | None, recorded: str | None = None) -> str:
        with mock.patch.object(
            runtime_env, "detect_max_compute_capability", return_value=capability
        ):
            return runtime_env.select_torch_index(recorded)

    def test_indexes_are_cuda_124_and_128(self) -> None:
        self.assertTrue(self.default_index.endswith("/whl/cu124"))
        self.assertTrue(self.blackwell_index.endswith("/whl/cu128"))

    def test_blackwell_gpus_use_cuda_128(self) -> None:
        self.assertEqual(self.select((12, 0)), self.blackwell_index)
        self.assertEqual(self.select((10, 0)), self.blackwell_index)

    def test_older_gpus_keep_cuda_124(self) -> None:
        self.assertEqual(self.select((9, 0)), self.default_index)
        self.assertEqual(self.select((8, 9)), self.default_index)
        self.assertEqual(self.select((7, 5)), self.default_index)

    def test_unknown_gpu_keeps_default_for_new_runtimes(self) -> None:
        self.assertEqual(self.select(None), self.default_index)

    def test_unknown_gpu_keeps_the_index_of_an_existing_runtime(self) -> None:
        self.assertEqual(self.select(None, self.blackwell_index), self.blackwell_index)
        self.assertEqual(self.select(None, self.default_index), self.default_index)

    def test_selection_is_logged_without_environment(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.select((12, 0))

        line = stderr.getvalue()
        self.assertIn("event.name=runtime.torch_index.selected", line)
        self.assertIn("torch.index=https://download.pytorch.org/whl/cu128", line)
        self.assertIn("device.compute_capability=12.0", line)

    def test_runtime_without_recorded_index_counts_as_default(self) -> None:
        with self.runtime(recorded=None) as root:
            status = runtime_env.runtime_status(root)

            self.assertEqual(status.torch_index, self.default_index)
            self.assertEqual(status.state, "ready")
            self.assertEqual(runtime_env.runtime_status(root, self.default_index).state, "ready")
            self.assertEqual(
                runtime_env.runtime_status(root, self.blackwell_index).state, "invalid"
            )

    def test_runtime_with_recorded_index_matches_only_that_index(self) -> None:
        with self.runtime(recorded=self.blackwell_index) as root:
            self.assertEqual(runtime_env.runtime_status(root).state, "ready")
            self.assertEqual(runtime_env.runtime_status(root, self.blackwell_index).state, "ready")
            self.assertEqual(runtime_env.runtime_status(root, self.default_index).state, "invalid")

    def test_ensure_runtime_rebuilds_default_runtime_for_blackwell_gpu(self) -> None:
        with self.runtime(recorded=None) as root:
            calls, written = self.ensure(root, capability=(12, 0))

        self.assertEqual(calls, [("create",), ("install", self.blackwell_index)])
        self.assertEqual(written["torch_index"], self.blackwell_index)

    def test_ensure_runtime_keeps_matching_runtime(self) -> None:
        with self.runtime(recorded=self.blackwell_index) as root:
            calls, written = self.ensure(root, capability=(12, 0))

        self.assertEqual(calls, [])
        self.assertEqual(written["torch_index"], self.blackwell_index)

    def test_ensure_runtime_keeps_default_runtime_for_older_gpu(self) -> None:
        with self.runtime(recorded=None) as root:
            calls, _written = self.ensure(root, capability=(8, 9))

        self.assertEqual(calls, [])

    def test_ensure_runtime_does_not_rebuild_when_probe_fails(self) -> None:
        with self.runtime(recorded=self.blackwell_index) as root:
            calls, written = self.ensure(root, capability=None)

        self.assertEqual(calls, [])
        self.assertEqual(written["torch_index"], self.blackwell_index)

    def test_ensure_runtime_rebuilds_for_different_gpu_generation(self) -> None:
        with self.runtime(recorded=self.blackwell_index) as root:
            calls, written = self.ensure(root, capability=(8, 6))

        self.assertEqual(calls, [("create",), ("install", self.default_index)])
        self.assertEqual(written["torch_index"], self.default_index)

    def test_index_change_is_logged(self) -> None:
        stderr = io.StringIO()
        with self.runtime(recorded=None) as root, contextlib.redirect_stderr(stderr):
            self.ensure(root, capability=(12, 0))

        line = stderr.getvalue()
        self.assertIn("event.name=runtime.torch_index.changed", line)
        self.assertIn("torch.index.previous=https://download.pytorch.org/whl/cu124", line)

    def test_install_uses_the_requested_torch_index(self) -> None:
        commands: list[list[str]] = []

        def fake_run_process(
            command: Sequence[str], _extra_env: Mapping[str, str], _context: object
        ) -> None:
            commands.append(list(command))

        with mock.patch.object(runtime_env, "_run_process", fake_run_process):
            runtime_env._install_dependencies(Path("python"), Path("cache"), None)
            runtime_env._install_dependencies(
                Path("python"), Path("cache"), None, self.blackwell_index
            )

        def index_of(command: list[str]) -> str:
            return command[command.index("--index-url") + 1]

        torch_installs = [command for command in commands if "--index-url" in command]
        self.assertEqual(
            [index_of(c) for c in torch_installs], [self.default_index, self.blackwell_index]
        )

    @contextlib.contextmanager
    def runtime(self, recorded: str | None) -> Iterator[Path]:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(runtime_env, "_python_version_matches_runtime", return_value=True),
        ):
            root = Path(temp_dir)
            python_path = runtime_env.venv_python(root / "venv")
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")
            (root / "venv" / "stale.txt").write_text("old", encoding="utf-8")
            marker: dict[str, object] = {
                "runtime_version": RUNTIME_VERSION,
                "python_version": RUNTIME_PYTHON_VERSION,
            }
            if recorded is not None:
                marker["torch_index"] = recorded
            (root / "runtime.json").write_text(json.dumps(marker), encoding="utf-8")
            yield root

    def ensure(
        self, root: Path, capability: tuple[int, int] | None
    ) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
        calls: list[tuple[str, ...]] = []
        ready = runtime_env.RuntimeReady(
            root=root,
            venv_path=root / "venv",
            python_executable=runtime_env.venv_python(root / "venv"),
            site_packages=root / "venv" / "site-packages",
            torch_version="2.8.0",
            cuda_available=True,
        )

        def fake_install(
            _python: Path, _cache: Path, _context: object, torch_index: str = TORCH_INDEX_URL
        ) -> None:
            calls.append(("install", torch_index))

        with (
            mock.patch.object(
                runtime_env, "detect_max_compute_capability", return_value=capability
            ),
            mock.patch.object(
                runtime_env, "_create_venv", lambda *_a, **_k: calls.append(("create",))
            ),
            mock.patch.object(runtime_env, "_install_dependencies", fake_install),
            mock.patch.object(runtime_env, "_validate_runtime", return_value=ready),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.dict(os.environ),
        ):
            runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)

        written: dict[str, Any] = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
        return calls, written


if __name__ == "__main__":
    unittest.main()
