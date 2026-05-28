from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.region_selection import RegionSelectionInput  # noqa: E402

from worker import selection_mask_builder  # noqa: E402


class SelectionMaskBuilderTests(unittest.TestCase):
    def test_best_mask_is_clipped_to_region_and_alpha(self) -> None:
        region = bytearray(16)
        for index in (5, 6, 9, 10):
            region[index] = 255
        alpha = bytearray(b"\xff" * 16)
        alpha[10] = 0
        selection_input = self._input(bytes(region), bytes(alpha))

        broad_mask = [[True for _x in range(4)] for _y in range(4)]
        result = selection_mask_builder.build_selection_mask([broad_mask], [0.9], selection_input)

        self.assertEqual(result.metadata[0]["area"], 3)
        self.assertEqual(result.selection_mask[5], 255)
        self.assertEqual(result.selection_mask[6], 255)
        self.assertEqual(result.selection_mask[9], 255)
        self.assertEqual(result.selection_mask[10], 0)

    def test_low_overlap_falls_back_to_region(self) -> None:
        region = bytearray(16)
        for index in (5, 6, 9, 10):
            region[index] = 255
        selection_input = self._input(bytes(region), b"\xff" * 16)
        empty_mask = [[False for _x in range(4)] for _y in range(4)]

        result = selection_mask_builder.build_selection_mask([empty_mask], [0.1], selection_input)

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.metadata[0]["area"], 4)

    @staticmethod
    def _input(region: bytes, alpha: bytes) -> RegionSelectionInput:
        return RegionSelectionInput(
            crop_image_array=object(),
            crop_alpha_mask=alpha,
            crop_region_mask=region,
            crop_bbox=(0, 0, 4, 4),
            region_bbox=(1, 1, 2, 2),
            prompt_box_xyxy=(1.0, 1.0, 3.0, 3.0),
            point_coords=((2.0, 2.0),),
            point_labels=(1,),
            canvas_width=4,
            canvas_height=4,
        )


if __name__ == "__main__":
    unittest.main()
