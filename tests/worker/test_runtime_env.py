from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from worker.runtime_manifest import RUNTIME_PYTHON_VERSION, RUNTIME_VERSION, SAM2_SOURCE_REF

from worker import runtime_env


class RuntimeEnvTests(unittest.TestCase):
    def test_runtime_status_reports_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = root / "venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            python_path.chmod(0o755)
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": RUNTIME_VERSION,
                        "python_version": f"{RUNTIME_PYTHON_VERSION}.13",
                        "sam2_source_ref": SAM2_SOURCE_REF,
                    }
                ),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "ready")
        self.assertEqual(status.version, RUNTIME_VERSION)
        self.assertEqual(status.sam2_source_ref, SAM2_SOURCE_REF)

    def test_runtime_status_reports_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = root / "venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            python_path.chmod(0o755)
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": "old",
                        "python_version": f"{RUNTIME_PYTHON_VERSION}.13",
                        "sam2_source_ref": SAM2_SOURCE_REF,
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
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            python_path.chmod(0o755)
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": RUNTIME_VERSION,
                        "python_version": f"{RUNTIME_PYTHON_VERSION}.13",
                        "sam2_source_ref": SAM2_SOURCE_REF,
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
                    site_packages=root / "venv" / "site-packages",
                    torch_version="2.5.1",
                    cuda_available=True,
                )

                ready = runtime_env.ensure_runtime(root, lambda _stage, _progress, _extra: None)
            finally:
                runtime_env._create_venv = original_create
                runtime_env._install_dependencies = original_install
                runtime_env._validate_runtime = original_validate

        self.assertEqual(calls, [])
        self.assertTrue(ready.cuda_available)

    def test_runtime_status_reports_invalid_python_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = root / "venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "runtime_version": RUNTIME_VERSION,
                        "python_version": "3.14.4",
                        "sam2_source_ref": SAM2_SOURCE_REF,
                    }
                ),
                encoding="utf-8",
            )

            status = runtime_env.runtime_status(root)

        self.assertEqual(status.state, "invalid")

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

    @staticmethod
    def _raise_bootstrap(*_args: Any, **_kwargs: Any) -> None:
        raise runtime_env.RuntimeBootstrapError("SAM2 runtime setup failed")

    def test_install_command_uses_uv_when_venv_has_no_pip(self) -> None:
        original_python_has_pip = runtime_env._python_has_pip
        original_which = runtime_env.shutil.which
        try:
            runtime_env._python_has_pip = lambda _python: False
            runtime_env.shutil.which = lambda name: "/usr/bin/uv" if name == "uv" else None

            command = runtime_env._install_command(
                Path("/tmp/runtime/venv/bin/python"),
                ("torch>=2.5.1",),
                index_url="https://example.invalid/simple",
            )
        finally:
            runtime_env._python_has_pip = original_python_has_pip
            runtime_env.shutil.which = original_which

        self.assertEqual(
            command,
            [
                "/usr/bin/uv",
                "pip",
                "install",
                "--python",
                "/tmp/runtime/venv/bin/python",
                "--index-url",
                "https://example.invalid/simple",
                "torch>=2.5.1",
            ],
        )

    def test_create_venv_uses_uv_for_missing_runtime(self) -> None:
        calls: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_which = runtime_env.shutil.which
            original_run_process = runtime_env._run_process
            try:
                runtime_env.shutil.which = lambda name: "/usr/bin/uv" if name == "uv" else None
                runtime_env._run_process = lambda args, _env, _context: calls.append(list(args))

                runtime_env._create_venv(root / "venv", None)
            finally:
                runtime_env.shutil.which = original_which
                runtime_env._run_process = original_run_process

        self.assertEqual(
            calls,
            [
                [
                    "/usr/bin/uv",
                    "venv",
                    "--python",
                    RUNTIME_PYTHON_VERSION,
                    "--clear",
                    str(root / "venv"),
                ]
            ],
        )


if __name__ == "__main__":
    unittest.main()
