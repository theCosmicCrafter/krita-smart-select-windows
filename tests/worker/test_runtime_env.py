from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.runtime_manifest import RUNTIME_PYTHON_VERSION, RUNTIME_VERSION  # noqa: E402

from worker import runtime_env  # noqa: E402


class RuntimeEnvTests(unittest.TestCase):
    def test_runtime_status_reports_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = root / "venv" / "bin" / "python"
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
            python_path = root / "venv" / "bin" / "python"
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
            python_path = root / "venv" / "bin" / "python"
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
                    python_executable=root / "venv" / "bin" / "python",
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


if __name__ == "__main__":
    unittest.main()
