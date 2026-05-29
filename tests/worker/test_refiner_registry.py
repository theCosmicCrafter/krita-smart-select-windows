from __future__ import annotations

import unittest

from worker.refiner_registry import (
    RefinerRegistryError,
    default_refiner_id,
    get_refiner_spec,
)


class RefinerRegistryTests(unittest.TestCase):
    def test_default_refiner_is_mam_vith(self) -> None:
        self.assertEqual(default_refiner_id(), "mam_sam_vith")

    def test_mam_vith_spec_has_exact_checkpoint_metadata(self) -> None:
        spec = get_refiner_spec("mam_sam_vith")

        self.assertEqual(spec.title, "MAM SAM ViT-H")
        self.assertEqual(
            spec.sam_checkpoint_url,
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        )
        self.assertEqual(spec.sam_checkpoint_filename, "sam_vit_h_4b8939.pth")
        self.assertEqual(spec.sam_expected_bytes, 2_564_550_879)
        self.assertIn("1fmBLZMWfGE2QOsc3pPqhzzlvBYojLBox", spec.mam_checkpoint_url)
        self.assertEqual(spec.mam_checkpoint_filename, "mam_vith.pth")
        self.assertEqual(spec.mam_expected_bytes, 32_785_332)

    def test_unknown_refiner_raises(self) -> None:
        with self.assertRaises(RefinerRegistryError):
            get_refiner_spec("mam_sam_vitb")


if __name__ == "__main__":
    unittest.main()
