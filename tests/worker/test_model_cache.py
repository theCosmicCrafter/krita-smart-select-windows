from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from email.message import Message
from io import StringIO
from pathlib import Path
from urllib.request import Request

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT_DIR / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from model_cache import CHUNK_SIZE, ModelCache, ModelDownloadCancelled  # noqa: E402
from model_registry import ModelSpec  # noqa: E402


def make_spec(expected_bytes: int = 12) -> ModelSpec:
    return ModelSpec(
        id="sam2.1_hiera_large",
        title="SAM2.1 Hiera Large",
        family="sam2.1",
        checkpoint_url="https://example.invalid/model.pt",
        checkpoint_filename="model.pt",
        config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
        expected_bytes=expected_bytes,
        etag='"etag"',
        recommended_ram_gb=32,
        recommended_vram_gb=12,
        profile="quality",
        requires_download=True,
    )


class FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class FakeResponseContext:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> FakeResponse:
        return self._response

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeOpener:
    def __init__(
        self,
        payload: bytes,
        *,
        range_supported: bool = True,
        ignore_first_range: bool = False,
    ) -> None:
        self.payload = payload
        self.range_supported = range_supported
        self.ignore_first_range = ignore_first_range
        self.calls: list[str | None] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponseContext:
        del timeout
        range_header = request.get_header("Range")
        self.calls.append(range_header)
        if self.ignore_first_range and len(self.calls) == 1:
            return FakeResponseContext(FakeResponse(self.payload))

        if not self.range_supported:
            return FakeResponseContext(FakeResponse(self.payload))

        if range_header is None:
            start = 0
            end = len(self.payload) - 1
        else:
            start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
            start = int(start_text)
            end = int(end_text) if end_text else len(self.payload) - 1
        return FakeResponseContext(
            FakeResponse(
                self.payload[start : end + 1],
                {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"},
            )
        )


class FlakyOpener(FakeOpener):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.failed_once = False

    def __call__(self, request: Request, *, timeout: float) -> FakeResponseContext:
        if not self.failed_once:
            self.failed_once = True
            raise TimeoutError("temporary stall")
        return super().__call__(request, timeout=timeout)


class ModelCacheTests(unittest.TestCase):
    def test_verified_file_with_exact_size_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = make_spec()
            model_dir = root / spec.id
            model_dir.mkdir()
            checkpoint = model_dir / "model.pt"
            checkpoint.write_bytes(b"abcdefghijkl")
            opener = FakeOpener(b"should-not-be-used")

            status = ModelCache(root, opener).ensure_downloaded(
                spec, lambda *_args: None, lambda: False
            )

        self.assertEqual(status.state, "verified")
        self.assertEqual(status.actual_bytes, 12)
        self.assertEqual(opener.calls, [])

    def test_missing_file_downloads_partial_then_renames_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = make_spec()
            progress: list[float | None] = []

            status = ModelCache(root, FakeOpener(b"abcdefghijkl")).ensure_downloaded(
                spec,
                lambda _stage, value, _extra: progress.append(value),
                lambda: False,
            )
            model_dir = root / spec.id
            sidecar = json.loads((model_dir / "model.json").read_text())
            self.assertEqual(status.state, "verified")
            self.assertEqual((model_dir / "model.pt").read_bytes(), b"abcdefghijkl")
            self.assertFalse((model_dir / "model.pt.partial").exists())
            self.assertEqual(sidecar["cache_state"], "verified")
            self.assertTrue(progress)

    def test_wrong_size_final_file_is_moved_to_invalid_and_redownloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = make_spec()
            model_dir = root / spec.id
            model_dir.mkdir()
            (model_dir / "model.pt").write_bytes(b"bad")

            status = ModelCache(root, FakeOpener(b"abcdefghijkl")).ensure_downloaded(
                spec, lambda *_args: None, lambda: False
            )
            self.assertEqual(status.state, "verified")
            self.assertEqual((model_dir / "model.pt.invalid").read_bytes(), b"bad")
            self.assertEqual((model_dir / "model.pt").read_bytes(), b"abcdefghijkl")

    def test_partial_file_resumes_when_range_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = make_spec()
            model_dir = root / spec.id
            model_dir.mkdir()
            (model_dir / "model.pt.partial").write_bytes(b"abcd")
            opener = FakeOpener(b"abcdefghijkl")

            status = ModelCache(root, opener).ensure_downloaded(
                spec, lambda *_args: None, lambda: False
            )
            self.assertEqual(status.state, "verified")
            self.assertEqual(opener.calls, ["bytes=4-11"])
            self.assertEqual((model_dir / "model.pt").read_bytes(), b"abcdefghijkl")

    def test_failed_resume_restarts_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = make_spec()
            model_dir = root / spec.id
            model_dir.mkdir()
            (model_dir / "model.pt.partial").write_bytes(b"abcd")
            opener = FakeOpener(b"abcdefghijkl", ignore_first_range=True)

            status = ModelCache(root, opener).ensure_downloaded(
                spec, lambda *_args: None, lambda: False
            )
            self.assertEqual(status.state, "verified")
            self.assertEqual(opener.calls, ["bytes=4-11", "bytes=0-11"])
            self.assertEqual((model_dir / "model.pt").read_bytes(), b"abcdefghijkl")

    def test_download_failure_logs_error_reason_to_stderr(self) -> None:
        def failing_opener(_request: Request, *, timeout: float) -> FakeResponseContext:
            del timeout
            raise TimeoutError("too slow")

        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = StringIO()
            with (
                redirect_stderr(stderr),
                self.assertRaisesRegex(Exception, "model download failed"),
            ):
                ModelCache(Path(temp_dir), failing_opener).ensure_downloaded(
                    make_spec(), lambda *_args: None, lambda: False
                )

        log = stderr.getvalue()
        self.assertIn("event=model.download.failed", log)
        self.assertIn("error_type=TimeoutError", log)
        self.assertIn("too slow", log)

    def test_protocol_download_failure_logs_error_reason_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = StringIO()
            with (
                redirect_stderr(stderr),
                self.assertRaisesRegex(Exception, "model download failed"),
            ):
                ModelCache(
                    Path(temp_dir), FakeOpener(b"abcdefghijkl", range_supported=False)
                ).ensure_downloaded(make_spec(), lambda *_args: None, lambda: False)

        log = stderr.getvalue()
        self.assertIn("event=model.download.failed", log)
        self.assertIn("error_type=ModelDownloadError", log)
        self.assertIn("missing content-range", log)

    def test_transient_download_failure_retries_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FlakyOpener(b"abcdefghijkl")
            status = ModelCache(Path(temp_dir), opener).ensure_downloaded(
                make_spec(), lambda *_args: None, lambda: False
            )

        self.assertEqual(status.state, "verified")
        self.assertEqual(opener.calls, ["bytes=0-11"])

    def test_cancellation_keeps_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = b"x" * (CHUNK_SIZE + 1)
            spec = make_spec(expected_bytes=len(payload))
            model_dir = root / spec.id
            cache = ModelCache(root, FakeOpener(payload))
            should_cancel = False

            def progress(_stage: str, _value: float | None, _extra: dict[str, object]) -> None:
                nonlocal should_cancel
                should_cancel = True

            with self.assertRaises(ModelDownloadCancelled):
                cache.ensure_downloaded(spec, progress, lambda: should_cancel)

            self.assertTrue((model_dir / "model.pt.partial").exists())
            self.assertFalse((model_dir / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
