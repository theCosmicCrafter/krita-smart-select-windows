from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT_DIR / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from model_registry import (  # noqa: E402
    SAM21_LARGE_CHECKPOINT_URL,
    SAM21_LARGE_EXPECTED_BYTES,
    ModelRegistryError,
    default_model_id,
    get_model_spec,
)


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_id_is_sam21_large(self) -> None:
        self.assertEqual(default_model_id(), "sam2.1_hiera_large")

    def test_large_spec_contains_official_checkpoint_metadata(self) -> None:
        spec = get_model_spec("sam2.1_hiera_large")

        self.assertEqual(spec.title, "SAM2.1 Hiera Large")
        self.assertEqual(spec.family, "sam2.1")
        self.assertEqual(spec.checkpoint_url, SAM21_LARGE_CHECKPOINT_URL)
        self.assertEqual(spec.checkpoint_filename, "sam2.1_hiera_large.pt")
        self.assertEqual(spec.config_name, "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.assertEqual(spec.expected_bytes, SAM21_LARGE_EXPECTED_BYTES)
        self.assertTrue(spec.requires_download)

    def test_mock_spec_requires_no_download(self) -> None:
        spec = get_model_spec("mock")

        self.assertEqual(spec.title, "Mock Segmentation")
        self.assertFalse(spec.requires_download)
        self.assertIsNone(spec.checkpoint_url)

    def test_unknown_model_raises_registry_error(self) -> None:
        with self.assertRaisesRegex(ModelRegistryError, "unknown model"):
            get_model_spec("unknown")


if __name__ == "__main__":
    unittest.main()
