from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.alpha_model_registry import get_model_spec  # noqa: E402

from worker import alpha_model_cache  # noqa: E402


class AlphaModelCacheTests(unittest.TestCase):
    def test_verified_snapshot_is_reused_without_network(self) -> None:
        spec = get_model_spec("birefnet_hr_matting")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / spec.id / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / spec.checkpoint_filename).write_bytes(b"0" * spec.checkpoint_expected_bytes)
            (snapshot / spec.config_filename).write_bytes(b"0" * spec.config_expected_bytes)
            cache = alpha_model_cache.ModelCache(root)

            status = cache.ensure_downloaded(spec, lambda *_args: None)

        self.assertEqual(status.state, "verified")
        self.assertEqual(status.checkpoint_bytes, spec.checkpoint_expected_bytes)

    def test_missing_snapshot_calls_downloader_with_pinned_repo(self) -> None:
        spec = get_model_spec("birefnet_hr_matting")
        calls: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = alpha_model_cache.ModelCache(root)

            def fake_snapshot_download(**kwargs: Any) -> None:
                calls.append(kwargs)
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / spec.checkpoint_filename).write_bytes(
                    b"0" * spec.checkpoint_expected_bytes
                )
                (local_dir / spec.config_filename).write_bytes(b"0" * spec.config_expected_bytes)

            original_import = alpha_model_cache.__dict__.get("__import__")
            try:
                alpha_model_cache.__dict__["__import__"] = lambda name: type(
                    "Hub",
                    (),
                    {"snapshot_download": staticmethod(fake_snapshot_download)},
                )()
                status = cache.ensure_downloaded(spec, lambda *_args: None)
            finally:
                if original_import is None:
                    alpha_model_cache.__dict__.pop("__import__", None)
                else:
                    alpha_model_cache.__dict__["__import__"] = original_import

        self.assertEqual(status.state, "verified")
        self.assertEqual(calls[0]["repo_id"], spec.repo_id)
        self.assertEqual(calls[0]["revision"], spec.revision)
        self.assertEqual(
            calls[0]["allow_patterns"], ["*.py", "*.json", "*.safetensors", "README.md"]
        )

    def test_wrong_size_checkpoint_invalidates_cache(self) -> None:
        spec = get_model_spec("birefnet_hr_matting")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / spec.id / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / spec.checkpoint_filename).write_bytes(b"too-small")
            (snapshot / spec.config_filename).write_bytes(b"0" * spec.config_expected_bytes)
            cache = alpha_model_cache.ModelCache(root)

            status = cache.status(spec)

        self.assertEqual(status.state, "invalid")

    def test_plugin_source_cache_path_is_rejected(self) -> None:
        spec = get_model_spec("birefnet_hr_matting")
        root = Path(__file__).resolve().parents[2] / "worker" / ".model-cache-test"
        cache = alpha_model_cache.ModelCache(root)

        with self.assertRaises(alpha_model_cache.ModelCacheDirMissing):
            cache.ensure_downloaded(spec, lambda *_args: None)


if __name__ == "__main__":
    unittest.main()
