from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

from worker.observability import log_event

PYTHON_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "python" / "smart_select"
if str(PYTHON_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))

from worker_launcher import format_log_event  # noqa: E402


class ObservabilityTests(unittest.TestCase):
    def capture_stderr(self, **fields: object) -> str:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            log_event("selection.start", **fields)
        return stderr.getvalue().strip()

    def test_log_event_emits_standard_fields(self) -> None:
        line = self.capture_stderr()

        self.assertIn("[SmartSelectWorker]", line)
        self.assertIn("event.name=selection.start", line)
        self.assertIn("event.domain=smart_select", line)
        self.assertIn("severity=info", line)
        self.assertIn("component=worker", line)

    def test_log_event_quotes_strings_with_spaces(self) -> None:
        line = self.capture_stderr(**{"error.message": "model download failed"})

        self.assertIn('error.message="model download failed"', line)

    def test_log_event_formats_json_like_values(self) -> None:
        line = self.capture_stderr(
            **{
                "value.none": None,
                "value.true": True,
                "value.false": False,
                "region.bbox": [1, 2, 3, 4],
            }
        )

        self.assertIn("value.none=null", line)
        self.assertIn("value.true=true", line)
        self.assertIn("value.false=false", line)
        self.assertIn("region.bbox=[1, 2, 3, 4]", line)

    def test_log_event_writes_to_stderr_only(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            log_event("model.download.failed", severity="error", component="model_cache")

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("event.name=model.download.failed", stderr.getvalue())
        self.assertIn("severity=error", stderr.getvalue())
        self.assertIn("component=model_cache", stderr.getvalue())

    def test_worker_launcher_uses_event_name(self) -> None:
        line = format_log_event(
            "worker.launcher.failed", **{"error.reason": "missing_worker_script"}
        )

        self.assertIn("event.name=worker.launcher.failed", line)
        self.assertIn("event.domain=smart_select", line)
        self.assertIn("component=worker_launcher", line)
        self.assertIn("error.reason=missing_worker_script", line)
        self.assertNotIn("event=worker.launcher.failed", line)


if __name__ == "__main__":
    unittest.main()
