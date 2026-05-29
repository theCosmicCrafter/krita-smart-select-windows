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

    def test_small_internal_holes_are_filled(self) -> None:
        selection_input = self._input(b"\xff" * 81, b"\xff" * 81, width=9, height=9)
        mask = [[True for _x in range(9)] for _y in range(9)]
        mask[4][4] = False

        result = selection_mask_builder.build_selection_mask([mask], [0.9], selection_input)

        self.assertEqual(result.metadata[0]["area"], 81)
        self.assertEqual(result.selection_mask[4 * 9 + 4], 255)

    def test_background_ring_candidate_is_penalized(self) -> None:
        selection_input = self._input(
            b"\xff" * 100,
            b"\xff" * 100,
            width=10,
            height=10,
            point_coords=((5.0, 5.0), (0.0, 0.0), (9.0, 0.0), (0.0, 9.0), (9.0, 9.0)),
            point_labels=(1, 0, 0, 0, 0),
        )
        object_mask = [[False for _x in range(10)] for _y in range(10)]
        background_ring = [[True for _x in range(10)] for _y in range(10)]
        for y in range(2, 8):
            for x in range(2, 8):
                object_mask[y][x] = True
                background_ring[y][x] = False

        result = selection_mask_builder.build_selection_mask(
            [background_ring, object_mask], [0.95, 0.6], selection_input
        )

        self.assertEqual(result.metadata[0]["area"], 36)
        self.assertEqual(result.selection_mask[5 * 10 + 5], 255)
        self.assertEqual(result.selection_mask[0], 0)

    def test_complementary_candidates_are_merged_for_recall(self) -> None:
        selection_input = self._input(
            b"\xff" * 80,
            b"\xff" * 80,
            width=10,
            height=8,
            point_coords=((2.0, 3.0), (7.0, 3.0)),
            point_labels=(1, 1),
        )
        left_half = [[x < 5 for x in range(10)] for _y in range(8)]
        right_half = [[x >= 5 for x in range(10)] for _y in range(8)]

        result = selection_mask_builder.build_selection_mask(
            [left_half, right_half], [0.9, 0.85], selection_input
        )

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.metadata[0]["area"], 80)
        self.assertEqual(result.selection_mask[3 * 10 + 2], 255)
        self.assertEqual(result.selection_mask[3 * 10 + 7], 255)

    def test_build_sam_prior_returns_crop_mask(self) -> None:
        selection_input = self._input(b"\xff" * 36, b"\xff" * 36, width=6, height=6)
        mask = [[True for _x in range(6)] for _y in range(6)]

        result = selection_mask_builder.build_sam_prior([mask], [0.9], selection_input)

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.crop_prior_mask, b"\xff" * 36)
        self.assertEqual(result.prior_area, 36)
        self.assertEqual(result.region_area, 36)
        self.assertEqual(result.region_coverage, 1.0)

    def test_build_selection_mask_from_crop_alpha_preserves_soft_values(self) -> None:
        region = bytearray(16)
        for index in (5, 6, 9, 10):
            region[index] = 255
        crop_alpha = bytearray(16)
        crop_alpha[5] = 64
        crop_alpha[6] = 128
        crop_alpha[9] = 255
        selection_input = self._input(bytes(region), b"\xff" * 16)

        result = selection_mask_builder.build_selection_mask_from_crop_alpha(
            bytes(crop_alpha), selection_input, 0.8
        )

        self.assertEqual(result.metadata[0]["area"], 3)
        self.assertEqual(result.selection_mask[5], 64)
        self.assertEqual(result.selection_mask[6], 128)
        self.assertEqual(result.selection_mask[9], 255)
        self.assertEqual(result.selection_mask[10], 0)

    @staticmethod
    def _input(
        region: bytes,
        alpha: bytes,
        *,
        width: int = 4,
        height: int = 4,
        point_coords: tuple[tuple[float, float], ...] = ((2.0, 2.0),),
        point_labels: tuple[int, ...] = (1,),
    ) -> RegionSelectionInput:
        return RegionSelectionInput(
            crop_image_array=object(),
            crop_alpha_mask=alpha,
            crop_region_mask=region,
            crop_bbox=(0, 0, width, height),
            region_bbox=(0, 0, width, height),
            prompt_box_xyxy=(0.0, 0.0, float(width), float(height)),
            point_coords=point_coords,
            point_labels=point_labels,
            canvas_width=width,
            canvas_height=height,
        )


if __name__ == "__main__":
    unittest.main()
