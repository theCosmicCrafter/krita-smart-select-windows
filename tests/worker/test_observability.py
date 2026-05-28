from __future__ import annotations

import contextlib
import io
import unittest

from worker.observability import log_event


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


if __name__ == "__main__":
    unittest.main()
