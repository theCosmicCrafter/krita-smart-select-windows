from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from urllib.request import Request

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT_DIR / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from refiner_cache import RefinerCache, RefinerDownloadCancelled  # noqa: E402
from refiner_registry import RefinerSpec, get_refiner_spec  # noqa: E402


class _Response:
    def __init__(self, payload: bytes, content_range: str) -> None:
        self._payload = payload
        self.headers = Message()
        self.headers["Content-Range"] = content_range

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


class _ResponseContext:
    def __init__(self, response: _Response) -> None:
        self._response = response

    def __enter__(self) -> _Response:
        return self._response

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object:
        return False


class _FakeOpener:
    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def __call__(self, request: Request, *, timeout: float) -> _ResponseContext:
        del timeout
        url = request.full_url
        self.calls.append(url)
        payload = self.payloads[url]
        range_header = request.get_header("Range") or f"bytes=0-{len(payload) - 1}"
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = int(end_text)
        chunk = payload[start : end + 1]
        return _ResponseContext(_Response(chunk, f"bytes {start}-{end}/{len(payload)}"))


class RefinerCacheTests(unittest.TestCase):
    def test_verified_cache_is_reused_without_network(self) -> None:
        spec = self._small_spec()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / spec.id
            cache_dir.mkdir()
            (cache_dir / spec.sam_checkpoint_filename).write_bytes(b"s" * spec.sam_expected_bytes)
            (cache_dir / spec.mam_checkpoint_filename).write_bytes(b"m" * spec.mam_expected_bytes)
            opener = _FakeOpener({})

            status = RefinerCache(root, opener).ensure_downloaded(
                spec, lambda _stage, _progress, _extra: None, lambda: False
            )

        self.assertEqual(status.state, "verified")
        self.assertEqual(opener.calls, [])

    def test_missing_cache_downloads_both_checkpoints(self) -> None:
        spec = self._small_spec()
        opener = _FakeOpener(
            {
                spec.sam_checkpoint_url: b"s" * spec.sam_expected_bytes,
                spec.mam_checkpoint_url: b"m" * spec.mam_expected_bytes,
            }
        )
        progress_stages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            status = RefinerCache(root, opener).ensure_downloaded(
                spec,
                lambda stage, _progress, _extra: progress_stages.append(stage),
                lambda: False,
            )

            self.assertEqual(status.state, "verified")
            self.assertTrue((root / spec.id / spec.sam_checkpoint_filename).exists())
            self.assertTrue((root / spec.id / spec.mam_checkpoint_filename).exists())
            self.assertTrue((root / spec.id / "refiner.json").exists())

        self.assertIn("downloading_refiner", progress_stages)
        self.assertEqual(len(opener.calls), 2)

    def test_wrong_size_checkpoint_is_invalid(self) -> None:
        spec = self._small_spec()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / spec.id
            cache_dir.mkdir()
            (cache_dir / spec.sam_checkpoint_filename).write_bytes(b"s" * spec.sam_expected_bytes)
            (cache_dir / spec.mam_checkpoint_filename).write_bytes(b"bad")

            status = RefinerCache(root).status(spec)

        self.assertEqual(status.state, "invalid")

    def test_cancellation_keeps_partial_file(self) -> None:
        spec = self._small_spec()
        opener = _FakeOpener(
            {
                spec.sam_checkpoint_url: b"s" * spec.sam_expected_bytes,
                spec.mam_checkpoint_url: b"m" * spec.mam_expected_bytes,
            }
        )
        calls = 0

        def is_cancelled() -> bool:
            nonlocal calls
            calls += 1
            return calls > 1

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(RefinerDownloadCancelled):
                RefinerCache(root, opener).ensure_downloaded(
                    spec, lambda _stage, _progress, _extra: None, is_cancelled
                )

            self.assertTrue((root / spec.id / f"{spec.sam_checkpoint_filename}.partial").exists())

    @staticmethod
    def _small_spec() -> RefinerSpec:
        original = get_refiner_spec("mam_sam_vith")
        return RefinerSpec(
            id=original.id,
            title=original.title,
            family=original.family,
            profile=original.profile,
            source_repo_url=original.source_repo_url,
            source_ref=original.source_ref,
            config_name=original.config_name,
            sam_model_type=original.sam_model_type,
            sam_checkpoint_url="https://example.invalid/sam.pth",
            sam_checkpoint_filename="sam.pth",
            sam_expected_bytes=4,
            sam_etag=None,
            mam_checkpoint_url="https://example.invalid/mam.pth",
            mam_checkpoint_filename="mam.pth",
            mam_expected_bytes=5,
            mam_etag=None,
            recommended_ram_gb=original.recommended_ram_gb,
            recommended_vram_gb=original.recommended_vram_gb,
            requires_download=True,
        )


if __name__ == "__main__":
    unittest.main()
