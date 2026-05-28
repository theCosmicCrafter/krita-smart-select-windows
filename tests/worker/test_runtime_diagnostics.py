from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT_DIR / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from runtime_diagnostics import collect_runtime_diagnostics  # noqa: E402


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_succeeds_without_torch(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("importlib.util.find_spec", return_value=None),
        ):
            diagnostics = collect_runtime_diagnostics(Path(temp_dir))

        self.assertEqual(diagnostics["torch"]["installed"], False)
        self.assertIn("executable", diagnostics["python"])
        self.assertIn("system", diagnostics["platform"])

    def test_diagnostics_reports_large_model_cache_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics = collect_runtime_diagnostics(Path(temp_dir))

        models = diagnostics["models"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "sam2.1_hiera_large")
        self.assertEqual(models[0]["cache_state"], "missing")
        self.assertEqual(models[0]["expected_bytes"], 898_089_053)

    def test_diagnostics_do_not_include_environment(self) -> None:
        diagnostics = collect_runtime_diagnostics(None)

        self.assertNotIn("env", diagnostics)
        self.assertNotIn("environment", diagnostics)


if __name__ == "__main__":
    unittest.main()
