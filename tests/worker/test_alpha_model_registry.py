from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.alpha_model_registry import (  # noqa: E402
    ModelRegistryError,
    default_model_id,
    get_model_spec,
)


class AlphaModelRegistryTests(unittest.TestCase):
    def test_default_model_is_birefnet_hr_matting(self) -> None:
        self.assertEqual(default_model_id(), "birefnet_hr_matting")

    def test_birefnet_spec_matches_pinned_metadata(self) -> None:
        spec = get_model_spec("birefnet_hr_matting")

        self.assertEqual(spec.title, "BiRefNet HR Matting")
        self.assertEqual(spec.repo_id, "ZhengPeng7/BiRefNet_HR-matting")
        self.assertEqual(spec.revision, "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f")
        self.assertEqual(spec.checkpoint_filename, "model.safetensors")
        self.assertEqual(spec.checkpoint_expected_bytes, 444_473_596)
        self.assertEqual(spec.config_filename, "config.json")
        self.assertEqual(spec.config_expected_bytes, 416)
        self.assertEqual(spec.license, "mit")
        self.assertEqual(spec.input_resolution, 2048)
        self.assertTrue(spec.trust_remote_code)

    def test_unknown_model_raises(self) -> None:
        with self.assertRaises(ModelRegistryError):
            get_model_spec("sam2.1_hiera_large")


if __name__ == "__main__":
    unittest.main()
