from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.region_selection import RegionSelectionInput  # noqa: E402

from worker import alpha_selection_builder  # noqa: E402


class AlphaSelectionBuilderTests(unittest.TestCase):
    def test_final_alpha_clips_to_lasso_region(self) -> None:
        region = self._region(crop_region_mask=bytes([0, 255, 255, 0]))

        result = alpha_selection_builder.build_selection_mask_from_crop_alpha(
            bytes([255, 128, 64, 255]), region
        )

        self.assertEqual(result.selection_mask, bytes([0, 128, 64, 0]))
        self.assertEqual(result.soft_pixels, 2)

    def test_final_alpha_clips_to_document_alpha(self) -> None:
        region = self._region(crop_alpha_mask=bytes([255, 0, 255, 0]))

        result = alpha_selection_builder.build_selection_mask_from_crop_alpha(
            bytes([255, 128, 64, 32]), region
        )

        self.assertEqual(result.selection_mask, bytes([255, 0, 64, 0]))

    def test_metadata_bbox_and_area_use_nonzero_alpha(self) -> None:
        region = RegionSelectionInput(
            crop_image_array=object(),
            crop_alpha_mask=b"\xff" * 4,
            crop_region_mask=b"\xff" * 4,
            crop_bbox=(1, 2, 2, 2),
            region_bbox=(1, 2, 2, 2),
            prompt_box_xyxy=(0.0, 0.0, 2.0, 2.0),
            point_coords=(),
            point_labels=(),
            canvas_width=5,
            canvas_height=5,
        )

        result = alpha_selection_builder.build_selection_mask_from_crop_alpha(
            bytes([0, 128, 0, 255]), region, score=1.0
        )

        self.assertEqual(result.metadata[0]["bbox"], [2, 2, 1, 2])
        self.assertEqual(result.metadata[0]["area"], 2)
        self.assertEqual(result.selection_mask[2 * 5 + 2], 128)
        self.assertEqual(result.selection_mask[3 * 5 + 2], 255)

    def test_empty_final_alpha_raises(self) -> None:
        with self.assertRaises(alpha_selection_builder.EmptySelectionError):
            alpha_selection_builder.build_selection_mask_from_crop_alpha(
                b"\x00" * 4, self._region()
            )

    @staticmethod
    def _region(
        crop_region_mask: bytes = b"\xff\xff\xff\xff",
        crop_alpha_mask: bytes = b"\xff\xff\xff\xff",
    ) -> RegionSelectionInput:
        return RegionSelectionInput(
            crop_image_array=object(),
            crop_alpha_mask=crop_alpha_mask,
            crop_region_mask=crop_region_mask,
            crop_bbox=(0, 0, 2, 2),
            region_bbox=(0, 0, 2, 2),
            prompt_box_xyxy=(0.0, 0.0, 2.0, 2.0),
            point_coords=(),
            point_labels=(),
            canvas_width=2,
            canvas_height=2,
        )


if __name__ == "__main__":
    unittest.main()
